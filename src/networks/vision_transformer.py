import torch
import torch.nn as nn

from functools import partial

from timm.models.layers import trunc_normal_, PatchEmbed

from src.networks.vision_transformer_layers import Block

def compute_grid_size(img_size, patch_size):
    if isinstance(img_size, int) and isinstance(patch_size, int):
        grid_size = img_size // patch_size
    else:
        if isinstance(img_size, (tuple, list)) and isinstance(patch_size, int):
            patch_size = [patch_size,] * len(img_size)
        elif isinstance(patch_size, (tuple, list)) and isinstance(img_size, int):
            img_size = [img_size,] * len(patch_size)
        grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            grid_size.append(im_size // pa_size)
    return grid_size


class ViT(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 use_learnable_pos_emb=False, return_hidden_states=False, return_cls_token=False,
                 pos_embed_builder=None, **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.return_hidden_states = return_hidden_states
        self.return_cls_token = return_cls_token
        self.return_attention = False  # Add this flag

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.return_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = 1 # no distill here
            # trunc_normal_(self.cls_token, std=.02, a=-.02, b=.02)
        else:
            self.cls_token = None
            self.num_tokens = 0
            
        grid_size = compute_grid_size(img_size, patch_size)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        else:
            assert pos_embed_builder is not None, \
                "When noting using learnable pos embed, pos embed builder should be specified"
            self.pos_embed = pos_embed_builder(grid_size, embed_dim, self.num_tokens)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            trunc_normal_(m, std=.02, a=-.02, b=.02)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        ret_hids = self.return_hidden_states
        ret_cls_token = self.return_cls_token
        
        x = self.patch_embed(x)
        
        if ret_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        if ret_hids:
            hidden_states_out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # print(f"num tokens after layer {i+1} is {x.size(1)}")
            if ret_hids:
                hidden_states_out.append(x[:, 1:, :])
        x = self.norm(x)
        
        if ret_cls_token:
            if ret_hids:
                return x[:, 0], hidden_states_out
            else:
                return x[:, 0], None
        else:
            if ret_hids:
                return x, hidden_states_out
            else:
                return x, None