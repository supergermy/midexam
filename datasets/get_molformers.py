import pandas as pd
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings

# Suppress the specific warning related to RobertaLMHeadModel
warnings.filterwarnings("ignore", message=".*RobertaForMaskedLM*")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

def get_molformers(task):
    """Convert SMILES to fingerprints and return as torch tensor"""
    if task == 'classification':
        tsv_path = '../datasets/B3DB_classification.tsv'
    elif task == 'regression':
        tsv_path = '../datasets/B3DB_regression.tsv'
    else:
        raise KeyError(f'task should be either "classification" or "regression", but got {task}')
    df = pd.read_csv(tsv_path, sep='\t')
    if task == 'classification':
        df['BBB+/BBB-'] = df['BBB+/BBB-'].map({'BBB-': 0, 'BBB+': 1})

    embd_tensor = torch.tensor([], dtype=torch.float32, device=device)

    with torch.no_grad():
        for smiles in tqdm(df['SMILES'], 'get_molformers'):
            padding=False
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=False).to(device)
            model_output = model(**encoded_input)
            embedding = model_output.pooler_output
            embd_tensor = torch.cat([embd_tensor,embedding], 0)
            
    return embd_tensor
