import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension


class FlexibleImageFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        train_test_dim: Optional[int] = None,
        class_dim: int = -1,
        domain_dim: Optional[int] = None,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
        ),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        extended_target: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train_test_dim = train_test_dim
        self.class_dim = class_dim
        self.domain_dim = domain_dim
        self.loader = loader
        self.extensions = extensions
        self.is_valid_file = is_valid_file
        self.extended_target = extended_target

        # Build the dataset
        self.samples, self.class_to_idx, self.domain_to_idx, self.train_test_to_idx = (
            self.make_dataset(self.root)
        )
        self.classes = list(self.class_to_idx.keys())
        self.domains = list(self.domain_to_idx.keys()) if self.domain_to_idx else []
        self.train_test_splits = (
            list(self.train_test_to_idx.keys()) if self.train_test_to_idx else []
        )
        self.targets = [s[1] for s in self.samples]

    def make_dataset(
        self, directory: str
    ) -> Tuple[
        List[Tuple[str, Dict[str, Any]]], Dict[str, int], Dict[str, int], Dict[str, int]
    ]:
        instances = []
        class_to_idx = {}
        domain_to_idx = {} if self.domain_dim is not None else None
        train_test_to_idx = {} if self.train_test_dim is not None else None

        # Walk through all files and collect samples
        for root_dir, _, file_names in os.walk(directory, followlinks=True):
            for fname in sorted(file_names):
                path = os.path.join(root_dir, fname)
                if self._is_valid_file(path):
                    # Split the path into components, excluding the filename
                    rel_path = os.path.relpath(path, directory)
                    rel_dir = os.path.dirname(rel_path)
                    parts = rel_dir.split(os.sep)

                    # Handle the case where rel_dir is empty
                    if parts == [""]:
                        parts = []

                    # Determine the maximum required depth
                    max_dim = max(
                        filter(
                            lambda x: x is not None,
                            [self.train_test_dim, self.class_dim, self.domain_dim],
                        )
                    )
                    if len(parts) <= max_dim:
                        warnings.warn(f"Not enough directory levels for file {path}")
                        continue

                    # Extract labels based on dimensions
                    class_name = parts[self.class_dim]
                    if class_name not in class_to_idx:
                        class_to_idx[class_name] = len(class_to_idx)
                    class_idx = class_to_idx[class_name]

                    domain_idx = -1
                    if self.domain_dim is not None:
                        domain_name = parts[self.domain_dim]
                        if domain_name not in domain_to_idx:
                            domain_to_idx[domain_name] = len(domain_to_idx)
                        domain_idx = domain_to_idx[domain_name]
                    else:
                        domain_name = None

                    train_test_idx = -1
                    if self.train_test_dim is not None:
                        train_test_name = parts[self.train_test_dim]
                        if train_test_name not in train_test_to_idx:
                            train_test_to_idx[train_test_name] = len(train_test_to_idx)
                        train_test_idx = train_test_to_idx[train_test_name]
                    else:
                        train_test_name = None

                    # Create the target tuple
                    target = {
                        "class_idx": class_idx,
                        "domain_idx": domain_idx,
                        "train_test_idx": train_test_idx,
                        "class_name": class_name,
                        "domain_name": domain_name,
                        "train_test_name": train_test_name,
                    }

                    instances.append((path, target))

        if not instances:
            raise FileNotFoundError(f"No valid data found in directory {directory}")

        return instances, class_to_idx, domain_to_idx, train_test_to_idx

    def _is_valid_file(self, path: str) -> bool:
        if self.is_valid_file is not None:
            return self.is_valid_file(path)
        if self.extensions is not None:
            return has_file_allowed_extension(path, self.extensions)
        else:
            return True  # Accept all files if no extensions or validation function is provided

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is a dictionary with labels.
        """
        path, target = self.samples[index]

        t = target.copy()
        if not self.extended_target:
            t = target["class_idx"]

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            # Apply target_transform to each target component if needed
            t = self.target_transform(t)

        return sample, t

    def __len__(self) -> int:
        return len(self.samples)

    def filter(
        self,
        train_test_name: Optional[str] = None,
        class_name: Optional[str] = None,
        domain_name: Optional[Union[str, List[str]]] = None,
    ):
        """
        Filters the dataset based on the provided criteria.

        Args:
            train_test_name (Optional[str]): Filter by train/test split name (e.g., 'train').
            class_name (Optional[str]): Filter by class name.
            domain_name (Optional[str]): Filter by domain name.

        Returns:
            None: Modifies the dataset in place.
        """
        filtered_samples = []
        if domain_name is not None and isinstance(domain_name, str):
            domain_name = [domain_name]

        for sample, target in self.samples:
            if (
                train_test_name is not None
                and target["train_test_name"] != train_test_name
            ):
                continue
            if class_name is not None and target["class_name"] != class_name:
                continue
            if domain_name is not None and target["domain_name"] not in domain_name:
                continue
            filtered_samples.append((sample, target))

        self.samples = filtered_samples
        self.targets = [target for _, target in self.samples]

        if not self.samples:
            warnings.warn(
                "No samples found after filtering with the provided criteria."
            )
        return self
