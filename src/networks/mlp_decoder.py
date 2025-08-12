import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Block


class MLPDecoder(nn.Module):
    def __init__(self, num_classes=768, embed_dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.head = nn.Linear(self.embed_dim, self.num_classes, bias=False) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.head(x)
        return x