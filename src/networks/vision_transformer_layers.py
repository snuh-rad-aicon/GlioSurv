from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import LayerScale

    
class PatchMerger(nn.Module):
    def __init__(self, in_dim, out_dim, num_tokens_out, bias=False):
        super(PatchMerger, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens_out = num_tokens_out
        self.queries = nn.Parameter(torch.randn(self.num_tokens_out, self.out_dim))
        self.kv = nn.Linear(in_dim, out_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        
    def forward(self, x):
        scale = self.out_dim ** -0.5
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        k, v = self.kv(x).split(self.out_dim, dim=-1)
        
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.out_dim)
        x = self.norm(x)
        return x.squeeze(1)


class DropoutPatchMerger(nn.Module):
    def __init__(self, in_dim, out_dim, num_tokens_out, bias=False, drop_rate=0.3):
        super(DropoutPatchMerger, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens_out = num_tokens_out
        self.drop_rate = drop_rate
        self.queries = nn.Parameter(torch.randn(self.num_tokens_out, self.out_dim))
        self.kv = nn.Linear(in_dim, out_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)

    def patch_merger(self, x):
        scale = self.out_dim ** -0.5
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        k, v = self.kv(x).split(self.out_dim, dim=-1)

        q = q * scale
        attn = q @ k.transpose(-2, -1)

        # apply dropkey
        mask = torch.ones_like(attn) * self.drop_rate
        attn = attn + torch.bernoulli(mask) * -1e12

        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.out_dim)
        x = self.norm(x)
        return x.squeeze(1)
        
    def forward(self, x):
        x1 = self.patch_merger(x)
        x2 = self.patch_merger(x)
        return x1, x2


class Attention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attention = False
        
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, attn.shape[-1])
            # attn_mask = attn_mask.unsqueeze(1) # (B, N, D)
            # attn_mask = attn_mask @ attn_mask.transpose(1, 2)
            attn = attn.masked_fill(~attn_mask, -1e9)
        
        attn = attn.softmax(dim=-1)
        attn_weights = attn
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.return_attention:
            return x, attn_weights
        return x
    
    
class AttentionLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        return self.attn(self.norm1(x), attn_mask)
    

class CrossAttention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = q_x.shape
        q = self.q(q_x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv_x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, attn.shape[-1])
            # attn_mask = attn_mask.unsqueeze(1) # (B, N, D)
            # attn_mask = attn_mask @ attn_mask.transpose(1, 2)
            attn = attn.masked_fill(~attn_mask, -1e9)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x1 = self.attn1(self.norm1(x1), self.norm2(x2), attn_mask)
        return x1

            
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    