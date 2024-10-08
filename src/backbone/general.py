from typing import Literal
import timm
from .adapter import add_adapters
from .ssf import add_ssf
from .vpt import add_vpt


def get_backbone(
    name, finetune_method: Literal["none", "adapter", "ssf", "vpt"] = "none", **kwargs
):
    model = timm.create_model(name, pretrained=True, num_classes=0)
    if finetune_method == "adapter":
        model = add_adapters(model, **kwargs)
    elif finetune_method == "ssf":
        model = add_ssf(model, **kwargs)
    elif finetune_method == "vpt":
        model = add_vpt(model, **kwargs)
    elif finetune_method == "none":
        # add the freeze method for consistency
        model.freeze = lambda fully=False: None
    return model
