import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_ecfp(smiles, mfgen):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return mfgen.GetFingerprintAsNumPy(molecule)

def get_fingerprints(task):
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

    fp_list = []
    label_list = []
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    for smiles, label in tqdm(zip(df['SMILES'], df['BBB+/BBB-']), desc='get_fingerprints', total=len(df)):
        fp = generate_ecfp(smiles, mfgen)
        fp_list.append(fp)
        label_list.append(label)
    
    # Convert to numpy array and handle any remaining None values
    desc_array = np.array(fp_list, dtype=np.float32)
    desc_array = np.nan_to_num(desc_array)  # Replace NaN with 0
    
    # Convert to torch tensor
    desc_tensor = torch.tensor(desc_array, dtype=torch.float32)
    
    label_array = np.array(label_list, dtype=np.float32)
    label_array = np.nan_to_num(label_array)  # Replace NaN with 0
    
    # Convert to torch tensor
    label_tensor = torch.tensor(label_array, dtype=torch.float32)

    return desc_tensor.to(device), label_tensor.to(device)
