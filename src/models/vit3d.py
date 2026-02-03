from matplotlib.pyplot import grid
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from timm.layers.helpers import to_3tuple

from src import networks
from src.networks.patch_embed_layers import PatchEmbed3D, build_3d_sincos_position_embedding


class ViT3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        
        encoder = getattr(networks, 'ViT')
        decoder = getattr(networks, 'MLPDecoder')
        
        encoder_embed_dim = 1024
        encoder_depth = 24
        encoder_num_heads = 16
        self.encoder = encoder(img_size=input_size,
                               patch_size=args.patch_size,
                               in_chans=args.in_chans,
                               embed_dim=encoder_embed_dim,
                               depth=encoder_depth,
                               num_heads=encoder_num_heads,
                               drop_path_rate=0.0,
                               embed_layer=PatchEmbed3D,
                               use_learnable_pos_emb=True,
                               return_hidden_states=False,
                               return_cls_token=True,
                               pos_embed_builder=build_3d_sincos_position_embedding,
        )
        self.decoder = decoder(num_classes=2,
                               embed_dim=encoder_embed_dim,
        )
    
    def get_num_layers(self):
        return {'encoder': self.encoder.get_num_layers()}

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.encoder: 'encoder',
                              self.decoder: 'decoder'}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay():
                    total_set.add(f'{prefix}.{name}')
        return total_set
    
    def forward(self, x_in):
        """
        x_in in shape of [BCHWD]
        """
        x, hidden_states = self.encoder(x_in)
        if len(x.size()) == 3:
            if x.size(1) == 1:
                x = x.squeeze(1)
            else:
                x = x.mean(dim=1)
                
        logits = self.decoder(x)
        return logits