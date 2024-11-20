import torch
from torch import nn
from timm.models.layers import Mlp
from typing import Any, Callable, Union, Optional, Tuple

from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    Attention,
    PatchEmbed,
)

from .util import convert_module, call_in_all_submodules


def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=0.02)
    nn.init.normal_(shift, std=0.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError(
            "the input tensor shape does not match the shape of the scale factor."
        )


class SSFModuleMinIn:
    def __init__(self, ssf_attr_list) -> None:
        self.ssf_attr_list = ssf_attr_list

        for attr_name in ssf_attr_list:
            attr = getattr(self, attr_name)
            # for each of these, init a scale and shift
            # first, get the dimension of the attribute
            dim = None
            if isinstance(attr, nn.Linear):
                dim = attr.out_features
            elif isinstance(attr, nn.LayerNorm):
                dim = attr.normalized_shape[0]
            elif isinstance(attr, nn.Conv2d):
                dim = attr.out_channels
            else:
                raise ValueError(f"Unsupported attribute type: {type(attr)}")

            # now init the params
            scale, shift = init_ssf_scale_shift(dim)
            setattr(self, f"{attr_name}_scale", scale)
            setattr(self, f"{attr_name}_shift", shift)

            # register hook that applies the ssf_ada function after the forward pass
            attr.register_forward_hook(self._get_ssf_hook(attr_name))

    def ssf_parameters(self):
        for attr_name in self.ssf_attr_list:
            scale = getattr(self, f"{attr_name}_scale")
            shift = getattr(self, f"{attr_name}_shift")
            yield scale
            yield shift

    def freeze(self, fully=False):
        for param in self.parameters():
            param.requires_grad = False
        if not fully:
            for param in self.ssf_parameters():
                param.requires_grad = True

        # call freeze on the children
        call_in_all_submodules(self, "freeze", fully=fully)

    def _get_ssf_hook(self, module_name):
        def _ssf_hook(module, input, output):
            scale = getattr(self, f"{module_name}_scale")
            shift = getattr(self, f"{module_name}_shift")
            output = ssf_ada(output, scale, shift)
            return output

        return _ssf_hook


class SSFAttention(Attention, SSFModuleMinIn):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0,
        proj_drop=0,
        norm_layer=nn.LayerNorm,
    ):
        Attention.__init__(
            self, dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer
        )
        SSFModuleMinIn.__init__(self, ["qkv", "proj"])


class SSFMlp(Mlp, SSFModuleMinIn):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0,
        use_conv=False,
    ):
        Mlp.__init__(
            self,
            in_features,
            hidden_features,
            out_features,
            act_layer,
            norm_layer,
            bias,
            drop,
            use_conv,
        )

        SSFModuleMinIn.__init__(self, ["fc1", "fc2"])


class SSFBlock(Block, SSFModuleMinIn):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0,
        attn_drop=0,
        init_values=None,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=...,
    ):
        Block.__init__(
            self,
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        SSFModuleMinIn.__init__(self, ["norm1", "norm2"])

        convert_module(
            self,
            {
                Attention: SSFAttention,
            },
            kwargs_calculation=_calculate_kwargs,
            info_assertion=_ssf_info_assertion,
        )


class SSFPatchEmbed(PatchEmbed, SSFModuleMinIn):
    def __init__(
        self,
        img_size: Union[int, None] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., Any]] = None,
        flatten: bool = True,
        output_fmt: Union[str, None] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        PatchEmbed.__init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            norm_layer,
            flatten,
            output_fmt,
            bias,
            strict_img_size,
            dynamic_img_pad,
        )
        ssfs = ["proj"]
        if norm_layer:
            ssfs.append("norm")
        SSFModuleMinIn.__init__(self, ssfs)


class SSFVisionTransformer(VisionTransformer, SSFModuleMinIn):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0,
        pos_drop_rate: float = 0,
        patch_drop_rate: float = 0,
        proj_drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        weight_init: str = "",
        embed_layer: Callable[..., Any] = SSFPatchEmbed,
        norm_layer: Union[str, Callable[..., Any], nn.Module, None] = None,
        act_layer: Union[str, Callable[..., Any], nn.Module, None] = None,
        block_fn: nn.Module = SSFBlock,
        mlp_layer: nn.Module = SSFMlp,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            pos_embed=pos_embed,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        SSFModuleMinIn.__init__(self, ["norm"] if use_fc_norm else [])


def _calualte_attention_kwargs(module, kwargs):
    if not isinstance(module, Attention):
        return kwargs
    # calculate dim and num_heads
    num_heads = module.num_heads
    dim = module.head_dim * num_heads

    # is a bool in the init, but is not stored as a bool
    qkv_bias = module.qkv.bias is not None
    return {"dim": dim, "num_heads": num_heads, "qkv_bias": qkv_bias, **kwargs}


def _calculate_kwargs(module, kwargs):
    if isinstance(module, Block):
        return _calualte_attention_kwargs(module.attn, kwargs)
    if isinstance(module, Attention):
        return _calualte_attention_kwargs(module, kwargs)
    return kwargs


def _ssf_info_assertion(info):
    valid = True

    # all missing_keys should end in _scale or _shift
    for key in info.missing_keys:
        valid = valid and (key.endswith("_scale") or key.endswith("_shift"))

    # no unexpected keys
    valid = valid and len(info.unexpected_keys) == 0

    return valid


def add_ssf(
    model: VisionTransformer,
):
    return convert_module(
        model,
        {
            VisionTransformer: SSFVisionTransformer,
        },
        info_assertion=_ssf_info_assertion,
        kwargs_calculation=_calculate_kwargs,
    )
