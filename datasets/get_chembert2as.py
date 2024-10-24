import pandas as pd
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import warnings

# Suppress the specific warning related to RobertaLMHeadModel
warnings.filterwarnings("ignore", message=".*RobertaForMaskedLM*")

device = "cuda" if torch.cuda.is_available() else "cpu"
chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(device)
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

def get_chembert2as(task):
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
        for smiles in tqdm(df['SMILES'], 'get_chembert2as'):
            padding=False
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True).to(device)
            model_output = chemberta(**encoded_input)
            embedding = model_output[0][:,0,:]
            embeddings_cls = embedding
            embd_tensor = torch.cat([embd_tensor,embeddings_cls], 0)
            
    return embd_tensor.view(-1,600)
