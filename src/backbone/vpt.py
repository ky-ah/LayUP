import torch
from torch import nn
from typing import Callable, Union, Optional, Tuple, Type, Literal

from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    PatchEmbed,
)
from timm.layers import Mlp, LayerType

from .util import convert_module


class VPTVisionTransformer(VisionTransformer):
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
        mlp_ratio: float = 4.0,
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
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        vpt_type: Literal["deep", "shallow"] = "deep",
        vpt_prompt_token_num: int = 10,
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

        # inistalize VPT
        self.vpt_type = vpt_type
        self.vpt_prompt_token_num = vpt_prompt_token_num

        vpt_shape_dict = {
            "shallow": (1, vpt_prompt_token_num, embed_dim),
            "deep": (depth, vpt_prompt_token_num, embed_dim),
        }

        self.vpt_prompt_tokens = nn.Parameter(
            torch.zeros(*vpt_shape_dict[vpt_type]), requires_grad=True
        )

        # register hooks for VPT
        self._register_vpt_hooks()

    def _register_vpt_hooks(self):
        if self.vpt_type == "deep":
            self._register_deep_vpt_hooks()
        else:
            self._register_shallow_vpt_hooks()

    def _register_deep_vpt_hooks(self):
        for i, block in enumerate(self.blocks):

            def pre_hook(module, args):
                # concat the input with the prompt tokens
                x = args[0]
                prompt_tokens = (
                    self.vpt_prompt_tokens[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                )
                x = torch.cat([x, prompt_tokens], dim=1)
                return (x, *args[1:])

            block.register_forward_pre_hook(
                hook=pre_hook,
                prepend=True,
            )

            def after_hook(module, args, output):
                # remove the prompt tokens from the output
                return output[:, : -self.vpt_prompt_token_num, :]

            block.register_forward_hook(
                hook=after_hook,
                prepend=True,  # so multi layer extraction won't be affected
            )

    def _register_shallow_vpt_hooks(self):
        # just one hook for all blocks
        def pre_hook(module, args):
            # concat the input with the prompt tokens
            x = args[0]
            prompt_tokens = self.vpt_prompt_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([x, prompt_tokens], dim=1)
            return (x, *args[1:])

        self.blocks.register_forward_pre_hook(
            hook=pre_hook,
            prepend=True,
        )

        def after_hook(module, args, output):
            # remove the prompt tokens from the output
            return output[:, : -self.vpt_prompt_token_num, :]

        self.blocks.register_forward_hook(
            hook=after_hook,
            prepend=True,  # so multi layer extraction won't be affected
        )

    def freeze(self, fully=False):
        for param in self.parameters():
            param.requires_grad = False
        if not fully:
            self.vpt_prompt_tokens.requires_grad = True


def _vpt_info_assert(info):
    valid = True

    # no unexpected keys
    valid = valid and not len(info.unexpected_keys)

    # missing_keys only contains the prompt tokens
    valid = valid and len(info.missing_keys) == 1
    valid = valid and info.missing_keys[0] == "vpt_prompt_tokens"

    return valid


def add_vpt(
    model: VisionTransformer,
    vpt_type: Literal["deep", "shallow"] = "deep",
    vpt_prompt_token_num: int = 10,
):
    return convert_module(
        model,
        cls_mapping={
            VisionTransformer: VPTVisionTransformer,
        },
        info_assertion=_vpt_info_assert,
        # kwargs
        vpt_type=vpt_type,
        vpt_prompt_token_num=vpt_prompt_token_num,
    )
