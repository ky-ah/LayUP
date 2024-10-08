import torch
from torch import nn
from typing import List
from tqdm import tqdm

from .modules import IntraLayerActivationWrapper, Ridge
from .backbone.util import call_in_all_submodules


class LayUP(nn.Module):
    def __init__(
        self,
        backbone,
        intralayers: List[str],
        num_classes,
        fast_ridge=True,
        ridge_bias=False,
    ) -> None:
        super().__init__()
        self.backbone = IntraLayerActivationWrapper(
            base_module=backbone,
            hooked_modules=intralayers,
        )
        self.intralayer_names = intralayers
        self.intralayer_names.sort()

        self.ridge = Ridge(
            in_features=self.backbone.num_features * len(intralayers),
            out_features=num_classes,
            fast=fast_ridge,
            bias=ridge_bias,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_backbone(self, input: torch.Tensor):
        """
        Forward the input through the backbone and store the intermediate activations
        Returns the final output and a dictionary of intermediate activations
        """
        # move input to the correct device
        input = input.to(self.device)
        # forward and store intermediate activations
        return self.backbone(input)

    def get_intra_activations(self, input: torch.Tensor):
        # forward and store activations
        _, hook_results = self.forward_backbone(input)
        # reformat into one tensor, based on self.intralayer_names order
        result = torch.cat(
            [hook_results[name] for name in self.intralayer_names], dim=1
        )
        return result

    def forward_with_fsa_head(self, input: torch.Tensor, head: nn.Module):
        """
        Forward the input through the backbone and the head
        """
        # forward and store intermediate activations
        backbone_output, _ = self.forward_backbone(input)
        # forward through the head
        output = head(backbone_output)
        return output

    def forward_with_ridge(self, input: torch.Tensor):
        """
        Forward the input through the backbone and the ridge
        """
        # get activations
        x = self.get_intra_activations(input)
        # forward through the ridge
        return self.ridge(x)

    @torch.no_grad()
    def update_ridge(
        self, dataloader, n_splits=4, possible_ridge_values=None, verbose=True
    ):
        all_activations = []
        all_labels = []

        data_iter = dataloader
        if verbose:
            data_iter = tqdm(data_iter, desc="Collecting activations")

        for batch in data_iter:
            input, label = batch
            all_activations.append(self.get_intra_activations(input).detach().cpu())
            all_labels.append(label)

        all_activations = torch.cat(all_activations, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.ridge.update(
            all_activations, all_labels, n_splits, possible_ridge_values, verbose
        )

    def forward(self, x):
        return self.forward_with_ridge(x)

    def freeze(self, fully=False):
        call_in_all_submodules(self, "freeze", fully=fully)
