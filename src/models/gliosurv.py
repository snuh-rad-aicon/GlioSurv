import torch
import torch.nn as nn

from timm.layers.helpers import to_3tuple

from src import networks
from src.networks.patch_embed_layers import PatchEmbed3D, build_3d_sincos_position_embedding
from src.data.mm_transforms import clinical_variable_token_list, clinical_variable_status_token_list, clinical_variable_treatment_token_list


class GlioSurv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        self.clinical_variable_token_list = clinical_variable_token_list
        self.clinical_variable_status_token_list = clinical_variable_status_token_list
        self.clinical_variable_treatment_token_list = clinical_variable_treatment_token_list
        vision_encoder = getattr(networks, 'ViT')
        
        vision_encoder_embed_dim = 1024
        vision_encoder_depth = 24
        vision_encoder_num_heads = 16
        self.vision_encoder = vision_encoder(img_size=input_size,
                               patch_size=args.patch_size,
                               in_chans=args.in_chans,
                               embed_dim=vision_encoder_embed_dim,
                               depth=vision_encoder_depth,
                               num_heads=vision_encoder_num_heads,
                               drop_path_rate=0.0,
                               embed_layer=PatchEmbed3D,
                               use_learnable_pos_emb=True,
                               return_hidden_states=False,
                               return_cls_token=True,
                               pos_embed_builder=build_3d_sincos_position_embedding,
        )
        self.language_features = nn.ModuleDict()
        loaded_features = getattr(networks, 'init_language_features')()
        for var_name, feature_dict in loaded_features.items():
            self.language_features[var_name] = nn.ParameterDict({
                str(k): nn.Parameter(v, requires_grad=True)
                for k, v in feature_dict.items()
            })
        self.text_encoder_embed_dim = 1024
        bottleneck_dim = 16
        decoder_embed_dim = 384
        decoder_depth = 2
        decoder_num_heads = 12
        decoder_mlp_ratio = 4.0
        decoder_qkv_bias = True
        decoder_drop_rate = 0.0
        decoder_attn_drop_rate = 0.0
        decoder_cross_attn_drop_rate = 0.1
        decoder_drop_path_rate = 0.0
        num_classes = 1
        num_tokens = 2
        max_encoder_length = 12
        mm_decoder = getattr(networks, 'TransformerDecoder')
        self.decoder = mm_decoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
            qkv_bias=decoder_qkv_bias,
            drop_rate=decoder_drop_rate,
            attn_drop_rate=decoder_attn_drop_rate,
            cross_attn_drop_rate=decoder_cross_attn_drop_rate,
            drop_path_rate=decoder_drop_path_rate,
            num_tokens=num_tokens,
            num_classes=num_classes,
            max_encoder_length=max_encoder_length,
        )
        self.vision_adapter = nn.Sequential(
            nn.Linear(vision_encoder_embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, decoder_embed_dim)
        )
        self.status_adapter = nn.Sequential(
            nn.Linear(self.text_encoder_embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, decoder_embed_dim)
        )
        self.treatment_adapter = nn.Sequential(
            nn.Linear(self.text_encoder_embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, decoder_embed_dim)
        )
        
        self.vision_encoder.eval()
        
    def set_train(self):
        self.decoder.train()
        self.vision_adapter.train()
        self.status_adapter.train()
        self.treatment_adapter.train()
        
    def set_eval(self):
        self.decoder.eval()
        self.vision_adapter.eval()
        self.status_adapter.eval()
        self.treatment_adapter.eval()

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.decoder: 'decoder'}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay():
                    total_set.add(f'{prefix}.{name}')
        return total_set
    
    def forward(self, x_in, clinival_variables, save_attn=False):
        vision_output = self.forward_vision(x_in)
        status_output = self.forward_language(clinival_variables, self.clinical_variable_status_token_list)
        treatment_output = self.forward_language(clinival_variables, self.clinical_variable_treatment_token_list)
        
        vision_output['features'] = self.vision_adapter(vision_output['features'])
        status_output['features'] = self.status_adapter(status_output['features'])
        treatment_output['features'] = self.treatment_adapter(treatment_output['features'])
        
        combined_features = torch.cat([vision_output['features'], status_output['features'], treatment_output['features']], dim=1)
        combined_attention_mask = torch.cat([vision_output['attention_mask'], status_output['attention_mask'], treatment_output['attention_mask']], dim=1)
        
        logits = self.decoder(encoder_out=combined_features, encoder_mask=combined_attention_mask, save_attn=save_attn)
        return logits
    
    def forward_vision(self, x_in):
        x, hidden_states = self.vision_encoder(x_in)
        x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        sequence_length = x.size(1)
        attention_mask = torch.ones(batch_size, sequence_length, device=x.device)
        
        return {
            'features': x,
            'attention_mask': attention_mask
        }
        
    def forward_language(self, clinival_variables, clinical_variable_token_list):
        device = next(self.parameters()).device
        batch_size = len(next(iter(clinival_variables.values())))
        num_variables = len(clinical_variable_token_list)
        
        clinical_features = torch.zeros(batch_size, num_variables, self.text_encoder_embed_dim, device=device)
        attention_mask = torch.zeros(batch_size, num_variables, dtype=torch.long, device=device)
        
        for var_idx, var_name in enumerate(clinical_variable_token_list):
            language_feature = torch.cat(
                [self.language_features[var_name][str(int(var_value))] for var_value in clinival_variables[var_name]],
                dim=0
            )
            language_mask = torch.where(
                torch.tensor([int(var_value) == -1 for var_value in clinival_variables[var_name]], device=device),
                torch.tensor(0, device=device),
                torch.tensor(1, device=device)
            )
            clinical_features[:, var_idx, :] = language_feature
            attention_mask[:, var_idx] = language_mask
            
        return {
            'features': clinical_features,
            'attention_mask': attention_mask
        }