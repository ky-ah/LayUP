import os
import copy
from typing import Tuple
from torchvision import datasets
from .flex import FlexibleImageFolder
from .download import download_domainnet


class AbstractImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root,
        train,
        transform=None,
        target_transform=None,
        train_subfolder="train",
        test_subfolder="test",
        root_folder="imagenetr",
    ):
        self.train = train

        if self.train:
            root = os.path.join(root, root_folder, train_subfolder)
        else:
            root = os.path.join(root, root_folder, test_subfolder)

        super().__init__(root, transform, target_transform)


class ImageNetR(AbstractImageFolder):
    # download here: https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR
    def __init__(self, root, train, transform=None, target_transform=None, postfix=""):
        super().__init__(
            root=root,
            train=train,
            root_folder="imagenetr",
            transform=transform,
            target_transform=target_transform,
            test_subfolder="test",
        )
        self.postfix = postfix

    def __str__(self) -> str:
        return "imagenetr" + self.postfix


class ImageNetA(AbstractImageFolder):
    # download here: https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p
    def __init__(
        self,
        root,
        train,
        transform=None,
        target_transform=None,
        train_subfolder="train",
        test_subfolder="test",
        root_folder="imagenet-a",
    ):
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            train_subfolder,
            test_subfolder,
            root_folder,
        )


class VTAB(AbstractImageFolder):
    # download here: https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_
    def __init__(self, root, train, transform=None, target_transform=None):
        super().__init__(
            root=root,
            train=train,
            root_folder="vtab",
            transform=transform,
            target_transform=target_transform,
            test_subfolder="test",
        )


class Cars(AbstractImageFolder):
    # download here: https://drive.google.com/file/d/1MbAlm4ciYNtWhMVL8K8_uxIcFtes2_jI
    def __init__(self, root, train, transform=None, target_transform=None):
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            train_subfolder="train",
            test_subfolder="test",
            root_folder="cars",
        )


class CUB(AbstractImageFolder):
    # download here: https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb
    def __init__(self, root, train, transform=None, target_transform=None):
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            train_subfolder="train",
            test_subfolder="test",
            root_folder="cub",
        )


class OmniBenchmark(AbstractImageFolder):
    # download here: https://drive.google.com/file/d/1GozYc4T5s3MkpEtYRoRhW92T-0sW4pFV
    def __init__(
        self,
        root,
        train,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root,
            train,
            transform,
            target_transform,
            root_folder="omnibenchmark",
        )


class CDDB(FlexibleImageFolder):
    # download here: https://coral79.github.io/CDDB_web/
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
    ):
        root = os.path.join(root, "CDDB")
        super().__init__(
            root=root,
            domain_dim=0,
            class_dim=-1,
            train_test_dim=1,
            transform=transform,
            target_transform=target_transform,
            extended_target=False,
        )

    def __str__(self) -> str:
        return "cddb"


class Domainnet(FlexibleImageFolder):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
    ):
        download_domainnet(root)
        root = os.path.join(root, "domainnet")
        super().__init__(
            root=root,
            domain_dim=0,
            class_dim=-1,
            train_test_dim=None,
            transform=transform,
            target_transform=target_transform,
            extended_target=False,
        )
        self._set_train_test(train)

    def _set_train_test(self, train: bool):
        # load train/test split txt files

        split_paths = set()

        for domain in self.domains:
            with open(
                os.path.join(self.root, f"{domain}_{'train' if train else 'test'}.txt"),
                "r",
            ) as f:
                # one line is relativa path [space] label
                domain_paths = f.read().splitlines()
                # get only the relative paths
                domain_paths = [line.split(" ")[0] for line in domain_paths]
                # convert to absolute paths
                domain_paths = [os.path.join(self.root, path) for path in domain_paths]

                split_paths.update(domain_paths)

        # filter samples
        self.samples = [s for s in self.samples if s[0] in split_paths]

    def __str__(self) -> str:
        return "domainnet"

    def __repr__(self) -> str:
        return f"Domainnet Dataset\nNumber of datapoints: {self.__len__()}"


class LimitedDomainnet(Domainnet):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        per_class_domain=10,
    ):
        super().__init__(root, train, transform, target_transform)
        # limit to per_class_domain samples per class per domain
        samples = copy.deepcopy(self.samples)
        self.samples = []

        for class_idx in range(len(self.classes)):
            for domain_idx in range(len(self.domains)):
                cur_samples = [
                    s
                    for s in samples
                    if s[1]["class_idx"] == class_idx
                    and s[1]["domain_idx"] == domain_idx
                ]
                self.samples.extend(cur_samples[:per_class_domain])


def imagenet_r(path="./data"):
    return (
        ImageNetR(root=path, train=True),
        ImageNetR(root=path, train=False),
    )


def imagenet_a(path="./data"):
    return (
        ImageNetA(root=path, train=True),
        ImageNetA(root=path, train=False),
    )


def dil_imagenet_r(path="./data"):
    return (
        ImageNetR(root=path, train=True, postfix="_dil"),
        ImageNetR(root=path, train=False, postfix="_dil"),
    )


def cifar100(path="./data"):
    return (
        datasets.cifar.CIFAR100(path, train=True, download=True),
        datasets.cifar.CIFAR100(path, train=False, download=True),
    )


def vtab(path="./data"):
    return (
        VTAB(root=path, train=True),
        VTAB(root=path, train=False),
    )


def cars(path="./data"):
    return (
        Cars(root=path, train=True),
        Cars(root=path, train=False),
    )


def cub(path="./data"):
    return (
        CUB(root=path, train=True),
        CUB(root=path, train=False),
    )


def omnibenchmark(path="./data"):
    return (
        OmniBenchmark(root=path, train=True),
        OmniBenchmark(root=path, train=False),
    )


def cddb(path="./data"):
    return (
        CDDB(root=path).filter(train_test_name="train"),
        CDDB(root=path).filter(train_test_name="val"),
    )


def domainnet(path="./data"):
    return (
        Domainnet(root=path, train=True),
        Domainnet(root=path, train=False),
    )


def limited_domainnet(path="./data", per_class_domain=10):
    return (
        LimitedDomainnet(root=path, train=True, per_class_domain=per_class_domain),
        Domainnet(root=path, train=False),
    )


DATASET_MAP = {
    # CIL
    "cifar100": cifar100,
    "imagenetr": imagenet_r,
    "imageneta": imagenet_a,
    "vtab": vtab,
    "cars": cars,
    "cub": cub,
    "omnibenchmark": omnibenchmark,
    # DIL
    "dil_imagenetr": dil_imagenet_r,
    "cddb": cddb,
    "limited_domainnet": limited_domainnet,
    # "domainnet": domainnet,
}


def get_dataset(
    dataset_name, path="./data"
) -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    return DATASET_MAP[dataset_name](path=path)
