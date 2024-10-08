from torchvision import transforms as transforms_module
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform


def update_transforms(
    dataset: VisionDataset, transforms=None, transform=None, target_transform=None
):
    has_transforms = transforms is not None
    has_separate_transform = transform is not None or target_transform is not None
    if has_transforms and has_separate_transform:
        raise ValueError(
            "Only transforms or transform/target_transform can be passed as argument"
        )

    # for backwards-compatibility
    dataset.transform = transform
    dataset.target_transform = target_transform

    if has_separate_transform:
        transforms = StandardTransform(transform, target_transform)
    dataset.transforms = transforms


def make_train_transform(
    target_size,
    resize_crop_min=0.999,
    resize_crop_max=1.0,
    random_rotation_degree=0,
    brightness_jitter=0.0,
    contrast_jitter=0.0,
    saturation_jitter=0.0,
    hue_jitter=0.0,
    normalize=False,
):
    t = []

    t.append(
        transforms_module.RandomResizedCrop(
            target_size,
            scale=(resize_crop_min, resize_crop_max),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
    )

    # Random horizontal flip
    t.append(transforms_module.RandomHorizontalFlip())

    t.append(
        transforms_module.ColorJitter(
            brightness=brightness_jitter,
            contrast=contrast_jitter,
            saturation=saturation_jitter,
            hue=hue_jitter,
        )
    )

    # Random rotation
    t.append(transforms_module.RandomRotation(random_rotation_degree))

    if normalize:
        t.append(
            transforms_module.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )

    t.append(transforms_module.ToTensor())

    return transforms_module.Compose(t)


def make_test_transform(
    target_size,
    normalize=False,
):
    t = []

    t.append(transforms_module.Resize(target_size))
    t.append(transforms_module.CenterCrop(target_size))

    if normalize:
        t.append(
            transforms_module.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )

    t.append(transforms_module.ToTensor())

    return transforms_module.Compose(t)


def make_train_transform_from_args(args):
    return make_train_transform(
        target_size=args.target_size,
        resize_crop_min=args.aug_resize_crop_min,
        resize_crop_max=args.aug_resize_crop_max,
        random_rotation_degree=args.aug_random_rotation_degree,
        brightness_jitter=args.aug_brightness_jitter,
        contrast_jitter=args.aug_contrast_jitter,
        saturation_jitter=args.aug_saturation_jitter,
        hue_jitter=args.aug_hue_jitter,
        normalize=args.aug_normalize,
    )


def make_test_transform_from_args(args):
    return make_test_transform(
        target_size=args.target_size,
        normalize=args.aug_normalize,
    )
