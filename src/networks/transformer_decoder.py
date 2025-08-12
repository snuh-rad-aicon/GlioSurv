import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_cross_attention=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.is_cross_attention = is_cross_attention

        if is_cross_attention:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn = None

    def forward(self, x, encoder_out=None, mask=None, encoder_mask=None, save_attn=False):
        B, N, C = x.shape
        
        if self.is_cross_attention:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            B_enc, N_enc, C_enc = encoder_out.shape
            kv = self.kv(encoder_out).reshape(B_enc, N_enc, 2, self.num_heads, C_enc // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if self.is_cross_attention and encoder_mask is not None:
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(1).float()  # float로 변환 필요
            attn = attn + (1.0 - encoder_mask) * -5.0  
        elif not self.is_cross_attention and mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).float()  # [B, 1, 1, N]
            attn = attn + (1.0 - mask) * -5.0

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if save_attn:
            attn.retain_grad()
            self.attn = attn
            
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):

        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., cross_attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        self.cross_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=cross_attn_drop, 
            proj_drop=drop, is_cross_attention=True)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, encoder_out=None, mask=None, encoder_mask=None, save_attn=False):
        x = x + self.drop_path(self.self_attn(self.norm1(x), mask=mask))
        
        if encoder_out is not None:
            x = x + self.drop_path(self.cross_attn(
                self.norm2(x),
                encoder_out=encoder_out,
                encoder_mask=encoder_mask,
                save_attn=save_attn
            ))
        
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True,
        drop_rate=0.0, 
        attn_drop_rate=0.0,
        cross_attn_drop_rate=0.1,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        num_classes=0,
        num_tokens=2,
        max_encoder_length=1024,
    ):

        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_tokens = num_tokens
        self.max_encoder_length = max_encoder_length
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.learnable_tokens = nn.Parameter(torch.zeros(num_tokens, embed_dim))
        nn.init.normal_(self.learnable_tokens, std=0.02)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, max_encoder_length, embed_dim))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                cross_attn_drop=cross_attn_drop_rate,
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        self.heads = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_tokens)])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, encoder_out=None, mask=None, encoder_mask=None, save_attn=False):

        batch_size = encoder_out.shape[0]
        encoder_seq_len = encoder_out.shape[1]
        
        tokens = self.learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = tokens + self.pos_embed
        
        if encoder_out is not None:
            encoder_pos_embed = self.encoder_pos_embed[:, :encoder_seq_len, :]
            encoder_out = encoder_out + encoder_pos_embed
        
        x = tokens
        
        for blk in self.blocks:
            x = blk(
                x,
                encoder_out=encoder_out,
                mask=mask,
                encoder_mask=encoder_mask,
                save_attn=save_attn
            )
        
        x = self.norm(x)
        x = torch.cat([self.heads[i](x[:, i, :]) for i in range(self.num_tokens)], dim=-1)
        
        return x