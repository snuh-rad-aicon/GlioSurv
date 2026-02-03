from re import L
import random
import numpy as np
import torch
from monai import transforms


clinical_variable_list = ['age', 'sex', 'who_grade', 'group', 'idh', '_1p19q', 'mgmt', 'kps', 
                          'eor_op1', 'radiotherapy1', 'chemotherapy1',
                          'age_description', 'sex_description', 'kps_description', 'who_grade_description', 'group_description', 'idh_description', '_1p19q_description', 'mgmt_description', 
                          'eor_op1_description', 'radiotherapy1_description', 'chemotherapy1_description',
                          'age_token_index', 'sex_token_index', 'kps_token_index', 'who_grade_token_index', 'group_token_index', 'idh_token_index', '_1p19q_token_index', 'mgmt_token_index', 
                          'eor_op1_token_index', 'radiotherapy1_token_index', 'chemotherapy1_token_index',
                          ]
clinical_variable_token_list = ['age', 'sex', 'who_grade', 'group', 'idh', '_1p19q', 'mgmt', 'kps', 
                          'eor_op1', 'radiotherapy1', 'chemotherapy1']
clinical_variable_status_token_list = ['age', 'sex', 'who_grade', 'group', 'idh', '_1p19q', 'mgmt', 'kps']
clinical_variable_treatment_token_list = ['eor_op1', 'radiotherapy1', 'chemotherapy1']

def map_age_category(age):
    if 15 <= age <= 47:
        return 0
    elif 48 <= age <= 63:
        return 1
    elif age >= 64:
        return 2
    else:
        return -1

def map_kps_category(kps):
    if 30 <= kps <= 50:
        return 0
    elif 60 <= kps <= 70:
        return 1
    elif 80 <= kps <= 100:
        return 2
    else:
        return -1
    
def map_who_grade_category(who_grade):
    if who_grade == 1:
        return 0
    elif who_grade == 2:
        return 1
    elif who_grade == 3:
        return 2
    elif who_grade == 4:
        return 3
    else:
        return -1
    
def map_group_category(group):
    if group == 1:
        return 0
    elif group == 2:
        return 1
    elif group == 3:
        return 2
    else:
        return -1
    
mapping_category_dict = {
    'age': map_age_category,
    'kps': map_kps_category,
    'who_grade': map_who_grade_category,
    'group': map_group_category,
}
    
def get_age_description(age):
    category = map_age_category(age)
    return age_mapping[category]
def get_kps_description(kps):
    category = map_kps_category(kps)
    return kps_mapping.get(category, "Performance status information is unavailable.")

sex_mapping = {0: "The patient is male.",
               1: "The patient is female."}

age_mapping = {
    0: "The patient is young.",
    1: "The patient is midlife.",
    2: "The patient is geriatric."
}

kps_mapping = {
    0: "The patient has a poor performance status.",
    1: "The patient has a moderate performance status.",
    2: "The patient has a good performance status."
}

who_grade_mapping = {
    0: "The tumor is classified as WHO grade I.",
    1: "The tumor is classified as WHO grade II.",
    2: "The tumor is classified as WHO grade III.",
    3: "The tumor is classified as WHO grade IV."
}

idh_mapping = {0: "The patient exhibits a mutant IDH genotype.",
            1: "The patient exhibits a wildtype IDH genotype."}

_1p19q_mapping = {
    0: "The patient exhibits a retained 1p19q genotype.",
    1: "The patient exhibits a deleted 1p19q genotype."
}

tumor_type_mapping = {
    0: "The tumor is a glioblastoma.",
    1: "The tumor is an astrocytoma.",
    2: "The tumor is an oligodendroglioma."
}

mgmt_mapping = {
    0: "The MGMT promoter of the tumor is unmethylated.",
    1: "The MGMT promoter of the tumor is methylated."
}

eor_op1_mapping = {
    0: "The patient underwent diagnostic surgery.",
    1: "The patient underwent partial surgery.",
    2: "The patient underwent complete surgery."
}

eor_op2_mapping = {
    0: "The patient underwent diagnostic follow-up surgery.",
    1: "The patient underwent partial follow-up surgery.",
    2: "The patient underwent complete follow-up surgery."
}

eor_op3_mapping = {
    0: "The patient underwent diagnostic third surgery.",
    1: "The patient underwent partial third surgery.",
    2: "The patient underwent complete third surgery."
}

radiotherapy1_mapping = { 
    0: "The patient shall not receive radiotherapy.",
    1: "The patient shall indeed receive radiotherapy.",
}

radiotherapy2_mapping = {
    0: "The patient shall not receive a second radiotherapy session.",
    1: "The patient shall indeed receive a second radiotherapy session.",
}

radiotherapy3_mapping = {
    0: "The patient shall not receive a third radiotherapy session.",
    1: "The patient shall indeed receive a third radiotherapy session.",
}

chemotherapy1_mapping = {
    0: "The patient shall not receive chemotherapy.",
    1: "The patient shall indeed receive chemotherapy.",
}

chemotherapy2_mapping = {
    0: "The patient shall not receive a second chemotherapy session.",
    1: "The patient shall indeed receive a second chemotherapy session.",
}

chemotherapy3_mapping = {
    0: "The patient shall not receive a third chemotherapy session.",
    1: "The patient shall indeed receive a third chemotherapy session.",
}

progression_event_mapping = {
    0: "Disease progression was absent in the patient.",
    1: "Disease progression was present in the patient."
}

mapping_dict = {
    "sex": sex_mapping,
    "age": age_mapping,
    "kps": kps_mapping,
    "who_grade": who_grade_mapping,
    "group": tumor_type_mapping,
    "idh": idh_mapping,
    "_1p19q": _1p19q_mapping,
    "mgmt": mgmt_mapping,
    "eor_op1": eor_op1_mapping,
    "eor_op2": eor_op2_mapping,
    "eor_op3": eor_op3_mapping,
    "radiotherapy1": radiotherapy1_mapping,
    "radiotherapy2": radiotherapy2_mapping,
    "radiotherapy3": radiotherapy3_mapping,
    "chemotherapy1": chemotherapy1_mapping,
    "chemotherapy2": chemotherapy2_mapping,
    "chemotherapy3": chemotherapy3_mapping,
    "progression_event": progression_event_mapping
}

CLS_TOKEN_OFFSET = 1

description_indices = {
    "sex": 3,
    "age": 3,
    "who_grade": 7,
    "group": 4,
    "idh": 4,
    "_1p19q": 4,
    "mgmt": 7,
    "eor_op1": 3,
    "eor_op2": 3,
    "eor_op3": 3,
    "radiotherapy1": 3,
    "radiotherapy2": 3,
    "radiotherapy3": 3,
    "chemotherapy1": 3,
    "chemotherapy2": 3,
    "chemotherapy3": 3,
    "progression_event": 3,
    "kps": 4
}

def get_tokenized_diff_index(text, diff_word, tokenizer):
    """
    Find the token index where the differing word starts after tokenization.

    Args:
        text (str): Full text string.
        diff_word (str): The word that differs.
        tokenizer: Tokenizer to use.

    Returns:
        int: Start index of the differing word after tokenization, or -1 if not found.
    """
    tokens = tokenizer.tokenize(text)
    for i, token in enumerate(tokens):
        # Subword tokenization may vary by tokenizer
        if diff_word in token or token in diff_word:
            return i
    return -1

def transform_label(label_dict):
    transformed = {}
    
    CLS_TOKEN_OFFSET = 1
    
    date_keys_to_drop = ['op1_date', 'op2_date', 'op3_date', 'death_or_last_fu_date', 
                         'death_date', 'progression_or_last_fu_date', 'progression_date']
    
    transformed = {}
       
    for key, value in label_dict.items():
        if key in date_keys_to_drop:
            continue
        
        if key in mapping_category_dict:
            value = mapping_category_dict[key](value)
        
        if isinstance(value, str):
            transformed[key] = value
        else:
            transformed[key] = torch.tensor(value)
                
        if key in clinical_variable_token_list:
            if value is not None and value != -1:
                if key in mapping_dict and value in mapping_dict[key]:
                    description = mapping_dict[key][value]
                    transformed[f'{key}_description'] = description
                    
                    if key in description_indices:
                        diff_index = description_indices[key]
                        transformed[f'{key}_token_index'] = diff_index + CLS_TOKEN_OFFSET
                        
            else:
                transformed[f'{key}_description'] = ""
                transformed[f'{key}_token_index'] = -1
        
    return transformed


