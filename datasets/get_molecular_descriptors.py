import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import torch
from tqdm import tqdm

import warnings
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

def get_molecular_descriptors(task):
    """Convert SMILES to molecular descriptors and return as torch tensor"""
    if task == 'classification':
        tsv_path = '../datasets/B3DB_classification.tsv'
    elif task == 'regression':
        tsv_path = '../datasets/B3DB_regression.tsv'
    else:
        raise KeyError(f'task should be either "classification" or "regression", but got {task}')
    df = pd.read_csv(tsv_path, sep='\t')
    if task == 'classification':
        df['BBB+/BBB-'] = df['BBB+/BBB-'].map({'BBB-': 0, 'BBB+': 1})

    desc_list = []
    calc = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

    for smiles in tqdm(df['SMILES'], 'get_molecular_descriptors'):
        mol = Chem.MolFromSmiles(smiles)
        desc = calc.CalcDescriptors(mol)
        desc_list.append(desc)
    
    # Convert to numpy array and handle any remaining None values
    desc_array = np.array(desc_list, dtype=np.float32)
    desc_array = np.nan_to_num(desc_array)  # Replace NaN with 0
    
    # Convert to torch tensor
    desc_tensor = torch.tensor(desc_array, dtype=torch.float32)
    
    return desc_tensor
