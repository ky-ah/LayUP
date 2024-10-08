from typing import Any, List
from torch import nn


class IntraLayerActivationWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, hooked_modules: List[str]) -> None:
        super().__init__()
        self.base_module = base_module
        self._hook_temp_storage = {}

        # register the hooks
        for module_name in hooked_modules:
            self.register_storage_hook(module_name)

    def register_storage_hook(self, module_name: str, cls_token_only=True):
        module = self.base_module
        for name in module_name.split("."):
            module = getattr(module, name)
        module.register_forward_hook(
            self._get_activation_hook(module_name, cls_token_only=cls_token_only)
        )

    def __getattr__(self, name: str) -> Any:
        """
        Will call base_module if the attribute is not found in the wrapper.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_module, name)

    def _get_activation_hook(self, name, cls_token_only=True):
        """
        Get the hook function
        Usage:
        ```
        self.clip_model.visual.transformer.resblocks[0]\
            .register_forward_hook(self._get_activation_hook('visual_transformer_resblocks0'))
        ```
        """

        def hook_cls(model, input, output):
            self._hook_temp_storage[name] = output[:, 0, :]

        def hook_all(model, input, output):
            self._hook_temp_storage[name] = output

        hook = hook_cls if cls_token_only else hook_all

        return hook

    def _copy_reset_hook_results(self):
        """
        Get the hook results and reset the temp storage
        """
        hook_res = {k: v for k, v in self._hook_temp_storage.items()}
        self._hook_temp_storage = {}
        return hook_res

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model
        Returns: Tuple
            result: Any (usually a tensor)
                The result of the forward pass
            hook_results: Dict
                The results of the hooks
        """
        self._hook_temp_storage = {}
        result = self.base_module(*args, **kwargs)
        hook_results = self._copy_reset_hook_results()
        return result, hook_results
