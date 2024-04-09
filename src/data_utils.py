# data_processing/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# data_processing/data_util.py
import torch
from torch.nn.functional import pad

def load_and_preprocess_data(data_path, sample_size=None, test_size=0.2, random_state=42):
    df_original = pd.read_csv(data_path)

    if sample_size:
        df = df_original.head(sample_size)
    else:
        df = df_original.copy()

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_data, test_data

def encode_labels(arg_relations):
    return LabelEncoder().fit_transform(arg_relations)



def get_distance_based_positional_embedding(distance, p_dim):
    # Calculate angles for sine and cosine
    angles = torch.arange(0, p_dim, 2.0) * -(1.0 / torch.pow(10000, (torch.arange(0.0, p_dim, 2.0) / p_dim)))

    # Calculate positional embeddings using sine and cosine functions
    positional_embeddings = torch.zeros(1, p_dim)
    positional_embeddings[:, 0::2] = torch.sin(distance * angles)
    positional_embeddings[:, 1::2] = torch.cos(distance * angles)

    return positional_embeddings

def tokenize_and_concat(propo1, propso2, argument, tokenizer):
    propos_arg = f"[BP1]  {propo1}  [EP1]  [BP2]  {propso2} [EP2]"
    propos= f"{propo1} ' [SEP] ' {propso2}"

    max_size = 256
    tokenized_input = tokenizer(propos, truncation=True, max_length=max_size, padding='max_length', return_tensors="pt")

    return tokenized_input,propos


def prepare_inputs(data, tokenizer):
    # Implementation of prepare_inputs
    tokenized_results = []
    position_embeddings = []
    micro_labels =  []
    input_data = []
    arguments = data['argument']
    prop_1_texts = data['prop_1']
    prop_2_texts = data['prop_2']
    for prop_1, prop_2, argument in zip(prop_1_texts, prop_2_texts, arguments):
        tokenized_input,propos = tokenize_and_concat(prop_1, prop_2, argument,tokenizer)
        tokenized_results.append(tokenized_input) 
        input_data.append(propos)
    return tokenized_results, input_data


