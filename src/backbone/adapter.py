import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
import timm
from functools import partial
from timm.models.vision_transformer import Block


from .util import convert_module


class Adapter(nn.Module):
    def __init__(
        self,
        n_embd=768,
        bottleneck=64,
        dropout=0.1,
        init_option="lora",
        scalar="0.1",
        layernorm_option="none",  # in, out, none
    ):
        super().__init__()
        self.n_embd = n_embd
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.init_option = init_option
        self.scalar = scalar
        self.layernorm_option = layernorm_option

        # _before

        self.adapter_layer_norm_before = None
        if layernorm_option == "in" or layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(scalar)

        self.down_proj = nn.Linear(self.n_embd, self.bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.bottleneck, self.n_embd)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        return up


class BlockWithAdapter(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        # adapter stuff
        adapter_bottleneck=64,
        adapter_drop=0.1,
        adapter_init_option="lora",
        adapter_scalar="0.1",
        adapter_layernorm_option="none",
        adapter_parallel=True,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        self.adapter_parallel = adapter_parallel

        self.adaptmlp = Adapter(
            n_embd=dim,
            bottleneck=adapter_bottleneck,
            dropout=adapter_drop,
            init_option=adapter_init_option,
            scalar=adapter_scalar,
            layernorm_option=adapter_layernorm_option,
        )

    @property
    def adapter_drop(self):
        return self.adaptmlp.dropout

    @property
    def adapter_init_option(self):
        return self.adaptmlp.init_option

    @property
    def adapter_scalar(self):
        return self.adaptmlp.scalar

    @property
    def adapter_layernorm_option(self):
        return self.adaptmlp.layernorm_option

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        if self.adapter_parallel:
            adapt_x = self.adaptmlp(x)

        residual = x

        x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if not self.adapter_parallel:
            adapt_x = x + self.adaptmlp(x)

        x = adapt_x + residual + x
        return x

    def freeze(self, fully=False):
        for param in self.parameters():
            param.requires_grad = False
        if not fully:
            for param in self.adaptmlp.parameters():
                param.requires_grad = True


def _calcualte_block_kwargs(module, kwargs):
    if not isinstance(module, Block):
        return kwargs
    # calculate dim and num_heads
    num_heads = module.attn.num_heads
    dim = module.attn.head_dim * num_heads

    # is a bool in the init, but is not stored as a bool
    qkv_bias = module.attn.qkv.bias is not None
    return {"dim": dim, "num_heads": num_heads, "qkv_bias": qkv_bias, **kwargs}


def _incompatible_keys_adapter_assertion(info):
    valid = True
    # missing_keys must only contain sting starting with "adaptmlp."
    missing_keys = info.missing_keys
    for mk in missing_keys:
        valid = valid and mk.startswith("adaptmlp.")
    # unexpected_keys must be empty
    valid = valid and len(info.unexpected_keys) == 0
    return valid


def add_adapters(
    model: timm.models.vision_transformer.VisionTransformer,
    adapter_bottleneck=64,
    adapter_drop=0.1,
    adapter_init_option="lora",
    adapter_scalar="0.1",
    adapter_layernorm_option="none",
    adapter_parallel=True,
):
    # add adpater to the model
    model = convert_module(
        model,
        {
            Block: partial(
                BlockWithAdapter,
                adapter_bottleneck=adapter_bottleneck,
                adapter_drop=adapter_drop,
                adapter_init_option=adapter_init_option,
                adapter_scalar=adapter_scalar,
                adapter_layernorm_option=adapter_layernorm_option,
                adapter_parallel=adapter_parallel,
            )
        },
        kwargs_calculation=_calcualte_block_kwargs,
        info_assertion=_incompatible_keys_adapter_assertion,
    )

    # freeze (will be unfozen)
    for param in model.parameters():
        param.requires_grad = False

    return model
