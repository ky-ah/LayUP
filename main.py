import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import functools

from torch.utils.data import DataLoader

from src.backbone import get_backbone
from src.modules import CosineLinear
from src.layup import LayUP
from src.data import (
    CILDataManager,
    DILDataManager,
    get_dataset,
    DATASET_MAP,
    make_test_transform_from_args,
    make_train_transform_from_args,
    update_transforms,
)
from src.logging import Logger, WandbLogger, ConsoleLogger, TQDMLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(args):
    Logger.instance().add_backend(ConsoleLogger())
    if args.wandb_project is not None:
        Logger.instance().add_backend(
            WandbLogger(args.wandb_project, args.wandb_entity, args)
        )


def update_args(args):
    assert args.k >= 1 and args.k <= 12
    args.intralayers = [f"blocks.{11 - i}" for i in range(args.k)]

    args.aug_normalize = bool(args.aug_normalize)

    args.target_size = 224

    return args


def fsa(model, train_dataset, test_dataset, args):
    model.freeze(fully=False)

    fsa_head = CosineLinear(
        in_features=model.backbone.num_features,
        out_features=train_dataset.num_classes,
        sigma=30,
    ).to(args.device)

    # set forward to use fsa head (instead of ridge)
    model.forward = functools.partial(model.forward_with_fsa_head, head=fsa_head)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters()},
            {"params": fsa_head.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=0.0
    )
    scheduler.last_epoch = -1

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # sanity
    # print all trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    for name, param in fsa_head.named_parameters():
        if param.requires_grad:
            print("head." + name)

    best_model_state_dict = None
    best_acc = 0.0
    epochs_no_improvement = 0

    for epoch in range(args.finetune_epochs):
        fsa_head.train()
        model.train()
        pbar = tqdm(dataloader, desc=f"Finetuning epoch {epoch}")
        Logger.instance().add_backend(TQDMLogger(pbar))
        for x, y in pbar:
            x = x.to(args.device)
            y = y.to(args.device)

            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            Logger.instance().log(
                {
                    "loss": loss.item(),
                    "tacc": (y_hat.argmax(1) == y).float().mean().item(),
                },
                blacklist_types=[ConsoleLogger],
            )

        scheduler.step()

        Logger.instance().pop_backend(TQDMLogger)
        eval_res = eval_dataset(model, test_dataset, args)
        acc = eval_res["acc"]
        eval_res = {"fsa_eval_" + k: v for k, v in eval_res.items()}

        if acc > best_acc:
            epochs_no_improvement = 0
            best_acc = acc
            best_model_state_dict = model.state_dict()
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= args.early_stopping:
            break

        Logger.instance().log(eval_res)

    # load best model
    model.load_state_dict(best_model_state_dict)

    # reset back to ridge forward
    model.forward = model.forward_with_ridge


def eval_datamanager(model, data_manager: CILDataManager, up_to_task: int, args):
    num_samples = {}
    results = {}
    for i, test_dataset in enumerate(data_manager.test_iter(up_to_task)):
        task_res = eval_dataset(model, test_dataset, args)
        results[i] = task_res
        num_samples[i] = len(test_dataset)

    # convert to formated string
    final_results = {
        "after_task": up_to_task,
    }
    for k, v in results.items():
        for kk, vv in v.items():
            final_results[f"task_{k}/{kk}"] = vv

    # add mean
    keys_to_mean = results[0].keys()
    for key in keys_to_mean:
        final_results[f"task_mean/{key}"] = np.mean(
            [v[key] for v in results.values()]
        ).item()

        # also use weighted mean
        final_results[f"task_wmean/{key}"] = np.average(
            [v[key] for v in results.values()],
            weights=[num_samples[k] for k in results],
        )

    return final_results


@torch.no_grad()
def eval_dataset(model, dataset, args):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    predictions = []
    labels = []

    for x, y in tqdm(dataloader, desc="Evaluating"):
        x = x.to(args.device)
        y = y.to(args.device)

        y_hat = model(x)

        predictions.append(y_hat.cpu().numpy())
        labels.append(y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    acc = (predictions.argmax(1) == labels).mean().item()

    return {"acc": acc}


def main(args):
    # get dataset and augmentations
    train_transform = make_train_transform_from_args(args)
    test_transform = make_test_transform_from_args(args)
    train_base_dataset, test_base_dataset = get_dataset(
        args.dataset, path=args.data_root
    )
    update_transforms(test_base_dataset, transform=test_transform)

    # get datamanager based on ds
    data_manager = None
    if DILDataManager.is_dil(str(train_base_dataset)):
        print("DIL")
        data_manager = DILDataManager(
            train_base_dataset,
            test_base_dataset,
        )
    else:
        print("CIL")
        data_manager = CILDataManager(
            train_base_dataset,
            test_base_dataset,
            T=args.T,
            num_first_task=None if args.dataset != "cars" else 16,
            shuffle=True,
            seed=args.seed,
        )
        # log datamanager info
        Logger.instance().log(
            {
                "class_order": data_manager.class_order,
            },
            blacklist_types=[WandbLogger],
        )

    Logger.instance().log(
        {
            "num_classes": data_manager.num_classes,
        }
    )

    # get model
    backbone = get_backbone(args.backbone, finetune_method=args.finetune_method)
    model = LayUP(
        backbone=backbone,
        intralayers=args.intralayers,
        num_classes=data_manager.num_classes,
    )
    model.to(args.device)

    for t, (train_dataset, test_datatset) in enumerate(data_manager):
        print(f"Task {t}")
        print(f"Train dataset: {len(train_dataset)}")
        print(f"Test dataset: {len(test_datatset)}")

        # first session adaptation
        if t == 0 and args.finetune_method != "none":
            train_dataset.transform = train_transform
            fsa(model, train_dataset, test_datatset, args)

        model.freeze(fully=True)
        model.eval()

        train_dataset.transform = test_transform

        # train Ridge regression
        model.update_ridge(
            DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        )

        # eval on all tasks up to t
        eval_res = eval_datamanager(model, data_manager, t, args)
        # log results
        Logger.instance().log(eval_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--early_stopping", type=int, default=5)

    # data
    parser.add_argument(
        "--dataset", type=str, default="cifar100", choices=DATASET_MAP.keys()
    )
    parser.add_argument("--T", type=int, default=10)

    # model
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_base_patch16_224",
        choices=["vit_base_patch16_224", "vit_base_patch16_224_in21k"],
    )
    parser.add_argument(
        "--finetune_method",
        type=str,
        default="none",
        choices=["none", "adapter", "ssf", "vpt"],
    )
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--k", type=int, default=6)

    # misc
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument(
        "--data_root",
        type=str,
        default="./local_data",
        help="Root directory for datasets",
    )

    # augmentations
    parser.add_argument("--aug_resize_crop_min", type=float, default=0.7)
    parser.add_argument("--aug_resize_crop_max", type=float, default=1.0)
    parser.add_argument("--aug_random_rotation_degree", type=int, default=0)
    parser.add_argument("--aug_brightness_jitter", type=float, default=0.1)
    parser.add_argument("--aug_contrast_jitter", type=float, default=0.1)
    parser.add_argument("--aug_saturation_jitter", type=float, default=0.1)
    parser.add_argument("--aug_hue_jitter", type=float, default=0.1)
    parser.add_argument("--aug_normalize", type=int, default=0)

    # logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    args = update_args(args)
    set_seed(args.seed)

    setup_logger(args)

    main(args)
