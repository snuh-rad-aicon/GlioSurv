import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModel

from src.data.mm_transforms import clinical_variable_token_list, mapping_dict, description_indices, CLS_TOKEN_OFFSET

def init_language_features():
    model_key = "bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    text_encoder = AutoModel.from_pretrained(model_key)

    language_features = {}
    for var in clinical_variable_token_list:
        current_mapping_dict = mapping_dict[var]
        token_index = description_indices[var] + CLS_TOKEN_OFFSET
        language_features[var] = {}
        for key, value in current_mapping_dict.items():
            feature = tokenizer(value, return_tensors="pt", padding=True, truncation=True)
            feature = text_encoder(**feature)
            feature = feature.last_hidden_state[0, token_index, :].unsqueeze(0)
            language_features[var][key] = feature
        language_features[var][-1] = torch.normal(
            mean=torch.zeros(1, 1024),
            std=torch.ones(1, 1024) * 0.02
        )

    return language_features
