import torch
from typing import Type, Dict
import inspect
from typing import List, Optional, Callable
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys


def get_extended_state_dict(module: torch.nn.Module, **extra_config) -> Dict[str, Dict]:
    """
    Get the extended state dict of a module
    """
    state_dict = module.state_dict()
    config_dict = {
        k: v
        for k, v in module.__dict__.items()
        if not k.startswith("_") and k != "training"
    }

    # Adding @properties to the config_dict
    for name, value in inspect.getmembers(
        type(module), lambda m: isinstance(m, property)
    ):
        config_dict[name] = getattr(module, name)

    return {"state_dict": state_dict, "config": {**config_dict, **extra_config}}


def load_from_extended_state_dict(
    module_cls: Type[torch.nn.Module],
    state_dict: Dict[str, Dict],
    strict: bool = False,
    **config_overrides,
):
    """
    Load a model from an extended state dict, adapting to the module class's constructor parameters
    """

    # Check if state_dict is extended
    if "config" not in state_dict:
        # We assume this will be a normal state dict
        # Allow for config overrides
        config = config_overrides
    else:
        # Merge configs from state dict and overrides
        config = {**state_dict["config"], **config_overrides}

    # Get the signature of the class constructor
    sig = inspect.signature(module_cls.__init__)
    params = sig.parameters

    # Check if **kwargs is accepted in the constructor
    supports_arbitrary_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if not supports_arbitrary_kwargs:
        # Filter out unsupported parameters
        filtered_config = {k: v for k, v in config.items() if k in params}
    else:
        # If **kwargs is supported, pass all config parameters
        filtered_config = config

    # Create the module instance with the filtered or full config
    module = module_cls(**filtered_config)

    # Load state dict appropriately based on the structure of the state dict provided
    if "config" in state_dict:
        info = module.load_state_dict(state_dict["state_dict"], strict=strict)
        return module, info
    else:
        info = module.load_state_dict(state_dict, strict=strict)
        return module, info


def _convert_single_module(
    module,
    cls_mapping,
    info_assertion: Optional[Callable[[_IncompatibleKeys], bool]] = None,
    kwargs_calculation: Optional[
        Callable[[nn.Module, Dict[str, any]], Dict[str, any]]
    ] = None,
    **kwargs,
):
    cls = cls_mapping.get(type(module), None)
    if cls is None:
        raise TypeError(
            f"Unsupported module {type(module)} for replacement, with mapping {cls_mapping}"
        )

    # get extended state dict form old module
    state_dict = get_extended_state_dict(module)

    if kwargs_calculation is not None:
        kwargs = kwargs_calculation(module, kwargs)
    # if hasattr(module, "bias") and "bias" not in kwargs:
    #     # (usually) bias is a bool in __init__ but a tensor in the state_dict
    #     kwargs["bias"] = module.bias is not None

    # Create a new instance with the same parameters
    new_module, info = load_from_extended_state_dict(
        cls, state_dict, strict=False, **kwargs
    )

    if info_assertion is not None:
        assert info_assertion(info), f"Assertion failed for info: {info}"

    return new_module


def _convert_modules_recusively(
    module: nn.Module,
    cls_mapping: Dict[Type[nn.Module], Type[nn.Module]],
    ignore: Optional[List[str]] = None,
    info_assertion: Optional[Callable[[_IncompatibleKeys], bool]] = None,
    kwargs_calculation: Optional[
        Callable[[nn.Module, Dict[str, any]], Dict[str, any]]
    ] = None,
    **kwargs,
):
    for name, sub_module in module.named_children():
        if ignore is not None and any(name in i for i in ignore):
            continue
        try:
            new_module = _convert_single_module(
                sub_module,
                cls_mapping,
                info_assertion,
                kwargs_calculation=kwargs_calculation,
                **kwargs,
            )
            setattr(module, name, new_module)
        except TypeError as e:
            if "for replacement, with mapping" not in str(e):
                raise e

            # if the module is not supported, try to convert the children
            _convert_modules_recusively(
                sub_module,
                cls_mapping,
                ignore=ignore,
                info_assertion=info_assertion,
                kwargs_calculation=kwargs_calculation,
                **kwargs,
            )

    return module


def _bias_kwargs_conversion(module, kwargs):
    """
    Convert the bias parameter from the module to the kwargs
    Used for Linear and Conv2d
    """
    if hasattr(module, "bias") and "bias" not in kwargs:
        # (usually) bias is a bool in __init__ but a tensor in the state_dict
        kwargs["bias"] = module.bias is not None
    return kwargs


def convert_module(
    module: nn.Module,
    cls_mapping: Dict[Type[nn.Module], Type[nn.Module]],
    ignore: Optional[List[str]] = None,
    info_assertion: Optional[Callable[[_IncompatibleKeys], bool]] = None,
    kwargs_calculation: Optional[
        Callable[[nn.Module, Dict[str, any]], Dict[str, any]]
    ] = None,
    recursive: bool = True,
    **kwargs,
):
    # if recursive is True, convert all children modules
    # but if the module itself has a mapping, convert
    can_convert_directly = type(module) in cls_mapping
    if recursive and not can_convert_directly:
        return _convert_modules_recusively(
            module,
            cls_mapping,
            ignore=ignore,
            info_assertion=info_assertion,
            kwargs_calculation=kwargs_calculation,
            **kwargs,
        )
    else:
        return _convert_single_module(
            module,
            cls_mapping,
            info_assertion=info_assertion,
            kwargs_calculation=kwargs_calculation,
            **kwargs,
        )


def call_in_all_submodules(module: nn.Module, def_name: str, **kwargs):
    """
    Calls a method in all submodules of a module and itself
    """
    # Recursively call the method in all submodules
    for _, sub_module in module.named_children():
        call_in_all_submodules(sub_module, def_name, **kwargs)
        # call the method itself
        if hasattr(sub_module, def_name):
            getattr(sub_module, def_name)(**kwargs)
