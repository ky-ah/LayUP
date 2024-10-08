from typing import Any, Tuple, Iterator, Optional
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import functools
from torchvision.datasets.vision import VisionDataset


def map_target(target, class_order_map):
    return class_order_map[target]


class SubDataset(Dataset):
    def __init__(
        self, dataset, incides, num_classes, transform=None, target_transfrom=None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = incides
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transfrom

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> Any:
        img, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CILDataManager:
    """
    T: number of tasks
    """

    def __init__(
        self,
        train_dataset: VisionDataset,
        test_dataset: VisionDataset,
        T: int = 10,
        num_first_task: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.T = T

        # find out how many classes there are
        # + dict of index => class
        uniue_targets = set()
        train_index_class_map = {}
        test_index_class_map = {}

        # get the right iterator
        train_target_iter = None
        test_target_iter = None

        def target_iter_regular(dataset):
            for _, target in dataset:
                yield target

        def target_iter_sample(dataset):
            # images are not loaded here -> faster
            for _, target in dataset.samples:
                yield target

        if hasattr(train_dataset, "samples"):
            train_target_iter = target_iter_sample(train_dataset)
            test_target_iter = target_iter_sample(test_dataset)
        else:
            train_target_iter = target_iter_regular(train_dataset)
            test_target_iter = target_iter_regular(test_dataset)

        # iterate over the datasets
        for i, target in tqdm(
            enumerate(train_target_iter), desc="Iterate over train dataset"
        ):
            uniue_targets.add(target)
            train_index_class_map[i] = target

        for i, target in tqdm(
            enumerate(test_target_iter), desc="Iterate over test dataset"
        ):
            test_index_class_map[i] = target

        self.num_classes = len(uniue_targets)

        # split classes into T tasks
        # if num_classes % T != 0, then the first task will have less classes
        effective_num_classes = (
            self.num_classes
            if num_first_task is None
            else self.num_classes - num_first_task
        )
        effective_T = T if num_first_task is None else T - 1
        classes_per_task = effective_num_classes // effective_T
        self.num_classes_per_task = [classes_per_task] * effective_T
        if num_first_task is not None:
            self.num_classes_per_task.insert(0, num_first_task)

        # sanity check
        assert sum(self.num_classes_per_task) == self.num_classes
        assert len(self.num_classes_per_task) == T

        # create a list of indices for each task
        self.train_indices_per_task = []
        self.test_indices_per_task = []
        self.class_order = list(uniue_targets)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            self.class_order = np.random.permutation(self.class_order).tolist()

        for i in range(T):
            start_idx = sum(self.num_classes_per_task[:i])
            end_idx = sum(self.num_classes_per_task[: i + 1])
            classes = set(self.class_order[start_idx:end_idx])

            train_indices = [
                idx
                for idx, target in train_index_class_map.items()
                if target in classes
            ]
            self.train_indices_per_task.append(train_indices)

            test_indices = [
                idx for idx, target in test_index_class_map.items() if target in classes
            ]
            self.test_indices_per_task.append(test_indices)

        # sanity check
        assert len(self.train_indices_per_task) == T
        assert len(self.test_indices_per_task) == T
        assert (
            sum(len(indices) for indices in self.train_indices_per_task)
            == len(self.train_dataset)
        ), f"{sum(len(indices) for indices in self.train_indices_per_task)} != {len(self.train_dataset)}"
        assert sum(len(indices) for indices in self.test_indices_per_task) == len(
            self.test_dataset
        )

    def __len__(self):
        return self.T

    def __getitem__(self, index) -> Tuple[SubDataset, SubDataset]:
        # map targets based on class order
        start_idx = sum(self.num_classes_per_task[:index])
        end_idx = sum(self.num_classes_per_task[: index + 1])
        class_order_map = {}

        for i, class_idx in enumerate(self.class_order[start_idx:end_idx]):
            class_order_map[class_idx] = i + start_idx

        target_transform = functools.partial(
            map_target, class_order_map=class_order_map
        )
        num_classes = len(class_order_map)
        return SubDataset(
            self.train_dataset,
            self.train_indices_per_task[index],
            num_classes=num_classes,
            target_transfrom=target_transform,
        ), SubDataset(
            self.test_dataset,
            self.test_indices_per_task[index],
            num_classes=num_classes,
            target_transfrom=target_transform,
        )

    def iter(self, up_to_task=None) -> Iterator[Tuple[SubDataset, SubDataset]]:
        up_to_task = self.T if up_to_task is None else up_to_task + 1
        for t in range(up_to_task):
            yield self[t]

    def test_iter(self, up_to_task=None) -> Iterator[SubDataset]:
        for _, test_dataset in self.iter(up_to_task):
            yield test_dataset


class DILDataManager:
    DIL_TASKS = {
        "cddb": [
            "gaugan",
            "biggan",
            "wild",
            "whichfaceisreal",
            "san",
        ],
        "domainnet": [
            "real",
            "quickdraw",
            "painting",
            "sketch",
            "infograph",
            "clipart",
        ],
        "imagenetr_dil": [
            "sketch",
            "art",
            "cartoon",
            "deviantart",
            "embroidery",
            "graffiti",
            "graphic",
            "misc",
            "origami",
            "painting",
            "sculpture",
            "sticker",
            "tattoo",
            "toy",
            "videogame",
        ],
    }

    @classmethod
    def is_dil(cls, dataset_name):
        return dataset_name in cls.DIL_TASKS

    def __init__(
        self,
        train_dataset: VisionDataset,
        test_dataset: VisionDataset,
    ) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.tasks = self.DIL_TASKS[str(self.train_dataset)]

        # count unique classes
        unique_classes = set()
        # create the index maps for each task
        self.train_indices_per_task = {}
        self.test_indices_per_task = {}
        for task in self.tasks:
            self.train_indices_per_task[task] = []
            self.test_indices_per_task[task] = []

            # check the paths of each image ans see if it contains the task name
            for i, (path, target) in enumerate(self.train_dataset.samples):
                if task in path:
                    self.train_indices_per_task[task].append(i)
                if isinstance(target, dict):
                    target = target["class_idx"]
                unique_classes.add(target)

            # do same for test set
            for i, (path, _) in enumerate(self.test_dataset.samples):
                if task in path:
                    self.test_indices_per_task[task].append(i)

        self.num_classes = len(unique_classes)

        # sanity check: no overlap
        task_index_sets = [
            set(indices) for indices in self.train_indices_per_task.values()
        ]
        assert len(set.intersection(*task_index_sets)) == 0, "Overlap in task indices"

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index) -> Tuple[SubDataset, SubDataset]:
        task = self.tasks[index]
        return SubDataset(
            self.train_dataset,
            self.train_indices_per_task[task],
            num_classes=self.num_classes,
        ), SubDataset(
            self.test_dataset,
            self.test_indices_per_task[task],
            num_classes=self.num_classes,
        )

    def iter(self, up_to_task=None) -> Iterator[Tuple[SubDataset, SubDataset]]:
        up_to_task = len(self) if up_to_task is None else up_to_task + 1
        for t in range(up_to_task):
            yield self[t]

    def test_iter(self, up_to_task=None) -> Iterator[SubDataset]:
        for _, test_dataset in self.iter(up_to_task):
            yield test_dataset
