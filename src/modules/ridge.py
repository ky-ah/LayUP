import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Optional, Iterable
from tqdm import tqdm
from ..logging.logger import Logger

DEFAULT_RIDGE_VALUES = np.logspace(-4, 3, num=15, base=10.0)
DEFAULT_RIDGE_VALUES = np.concatenate([DEFAULT_RIDGE_VALUES, [1e-8]])


def target2onehot(targets, n_classes):
    """
    Converts a batch of target labels into a one-hot encoded tensor.

    Parameters:
    targets (torch.Tensor): Tensor of target labels with shape (batch_size,).
    n_classes (int): Number of classes.

    Returns:
    torch.Tensor: One-hot encoded tensor with shape (batch_size, n_classes).
    """
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def calculate_ridge_weights(G: torch.Tensor, c: torch.Tensor, ridge: float):
    """
    Solves the ridge regression weights using the closed-form solution.

    Parameters:
    G (torch.Tensor): Gram matrix of features with shape (n_features, n_features).
    c (torch.Tensor): Target matrix with shape (n_classes, n_features).
    ridge (float): Regularization strength.

    Returns:
    torch.Tensor: Weight matrix with shape (n_classes, n_features).
    """
    return torch.linalg.solve(G + ridge * torch.eye(G.size(dim=0), device=G.device), c)


@torch.no_grad()
def optimize_ridge(
    features: torch.FloatTensor,
    labels: torch.LongTensor,
    one_hot_labels: Optional[torch.FloatTensor] = None,
    n_splits: int = 4,
    G: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    possible_ridge_values: Optional[Iterable] = None,
    verbose=False,
):
    """
    Optimizes the ridge regression regularization parameter using cross-validation.

    Parameters:
    features (torch.FloatTensor): Feature matrix with shape (n_samples, n_features).
    labels (torch.LongTensor): Tensor of target labels with shape (n_samples,).
    one_hot_labels (Optional[torch.FloatTensor]): One-hot encoded labels (default: None).
    n_splits (int): Number of cross-validation folds (default: 4).
    G (Optional[torch.Tensor]): Precomputed Gram matrix (default: None).
    c (Optional[torch.Tensor]): Precomputed target matrix (default: None).
    possible_ridge_values (Optional[Iterable]): Set of ridge values to test (default: None).
    verbose (bool): Whether to print progress information (default: False).

    Returns:
    float: The best ridge value based on cross-validation accuracy.
    """
    num_classes = labels.max() + 1

    # set defaults
    if possible_ridge_values is None:
        possible_ridge_values = DEFAULT_RIDGE_VALUES

    if one_hot_labels is None:
        one_hot_labels = target2onehot(labels, num_classes)

    # prepare for cross-validation
    global_accs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    splits = list(skf.split(features, labels.to("cpu").numpy()))
    if verbose:
        splits = tqdm(splits, desc="Optimizing Ridge")

    for train_idx, val_idx in splits:
        train_features = features[train_idx]
        val_features = features[val_idx]
        train_labels = one_hot_labels[train_idx]
        val_labels = one_hot_labels[val_idx]

        # calculate the Gram matrix
        train_G = train_features.T @ train_features
        if G is not None:
            train_G = train_G.to(G.device)
            train_G += G
            train_G = train_G.cpu()

        # calculate the target matrix
        train_c = train_features.T @ train_labels
        if c is not None:
            train_c = train_c.to(c.device)
            train_c += c
            train_c = train_c.cpu()

        # calculate the accuracy
        ridge_accs = []
        for ridge in possible_ridge_values:
            weights = calculate_ridge_weights(train_G, train_c, ridge)
            preds = val_features @ weights
            preds = preds.argmax(dim=1)
            acc = (preds == val_labels.argmax(dim=1)).float().mean().item()
            ridge_accs.append(acc)

        global_accs.append(ridge_accs)

    # calculate the mean accuracy
    # and select the best ridge
    global_accs = np.array(global_accs)
    mean_accs = global_accs.mean(axis=0)
    ridge = possible_ridge_values[np.argmax(mean_accs)]

    if verbose:
        # convert possible_ridge_values and mean_accs to list
        possible_ridge_values = possible_ridge_values.tolist()
        mean_accs = mean_accs.tolist()

        # loggin the possible_ridge_values and mean_accs
        possible_ridge_values = [
            "Ridge:" + str(ridge) for ridge in possible_ridge_values
        ]
        ridge_dict = dict(zip(possible_ridge_values, mean_accs))

        # add the selected ridge to the ridge_dict
        ridge_dict["selected_ridge"] = ridge

        # log the ridge_dict
        Logger.instance().log(ridge_dict)

    return ridge


class Ridge(nn.Module):
    """
    Parameters:
    in_features (int): Dimensionality of the input features.
    out_features (int): Number of output classes.
    fast (bool): Whether to precompute the weights for fast inference (default: True).
    bias (bool): Whether to include a bias term in the model (default: False).
    """

    def __init__(
        self, in_features: int, out_features: int, fast: bool = True, bias: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.register_buffer(
            "G", torch.zeros(self.effective_in_features, self.effective_in_features)
        )
        self.register_buffer("c", torch.zeros(self.effective_in_features, out_features))
        self.ridge = 0

        if fast:
            self.register_buffer(
                "weight", torch.zeros(out_features, self.effective_in_features)
            )
        else:
            self.register_buffer("weight", None)

    @property
    def effective_in_features(self):
        """
        Computes the effective dimensionality of the model, taking into account the bias term.

        Returns:
        int: Effective dimension (dim + 1 if bias is included, otherwise dim).
        """
        return self.in_features + 1 if self.bias else self.in_features

    def potentially_add_bias(self, features):
        """
        Adds a bias column to the feature matrix if necessary.

        Parameters:
        features (torch.Tensor): Feature matrix with shape (n_samples, dim).

        Returns:
        torch.Tensor: Feature matrix with bias column added if required.
        """
        if self.bias and features.size(1) == self.in_features:
            return torch.cat(
                [features, torch.ones(features.size(0), 1, device=features.device)],
                dim=1,
            )
        return features

    def update(
        self, features, labels, n_splits=4, possible_ridge_values=None, verbose=False
    ):
        """
        Updates the model with new data, optimizing the ridge parameter and updating weights.

        Parameters:
        features (torch.Tensor): Feature matrix with shape (n_samples, dim).
        labels (torch.LongTensor): Tensor of target labels with shape (n_samples,).
        n_splits (int): Number of cross-validation folds (default: 4).
        possible_ridge_values (Optional[Iterable]): Set of ridge values to test (default: None).
        verbose (bool): Whether to print progress information (default: False).
        """
        features = self.potentially_add_bias(features)

        # calculate the one-hot labels
        one_hot_labels = target2onehot(labels, self.out_features)

        # optimize the ridge
        self.ridge = optimize_ridge(
            features,
            labels,
            one_hot_labels=one_hot_labels,
            n_splits=n_splits,
            G=self.G,
            c=self.c,
            possible_ridge_values=possible_ridge_values,
            verbose=verbose,
        )
        # update the G and c matrices
        # and possibly the weights
        self.G += (features.T @ features).to(self.G.device)
        self.c += (features.T @ one_hot_labels).to(self.c.device)
        if self.weight is not None:
            self.weight = calculate_ridge_weights(self.G, self.c, self.ridge)

    def forward(self, x):
        """
        Performs a forward pass through the model, computing the output predictions.

        Parameters:
        x (torch.Tensor): Input feature matrix with shape (n_samples, dim).

        Returns:
        torch.Tensor: Output predictions with shape (n_samples, num_classes).
        """
        # add bias if necessary
        x = self.potentially_add_bias(x)

        # calculate the output
        # if the weight is already calculated, use it
        if self.weight is not None:
            return x @ self.weight
        else:
            return x @ calculate_ridge_weights(self.G, self.c, self.ridge)
