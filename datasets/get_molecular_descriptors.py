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
    """Convert SMILES to molecular descriptors and return normalized torch tensor"""
    # Define the descriptors we want to keep
    selected_descriptors = [
        'MolMR', 'fr_Ar_N', 'SMR_VSA3', 'VSA_EState5', 'fr_isothiocyan',
        'fr_C_O_noCOO', 'fr_SH', 'FpDensityMorgan1', 'SlogP_VSA12',
        'MinAbsEStateIndex', 'PEOE_VSA13', 'fr_NH0', 'MaxEStateIndex',
        'SlogP_VSA4', 'SMR_VSA4', 'NumSaturatedCarbocycles', 'fr_sulfide',
        'fr_sulfone', 'Chi0', 'fr_epoxide'
    ]
    
    if task == 'classification':
        tsv_path = '../datasets/B3DB_classification.tsv'
    elif task == 'regression':
        tsv_path = '../datasets/B3DB_regression.tsv'
    else:
        raise KeyError(f'task should be either "classification" or "regression", but got {task}')
    
    df = pd.read_csv(tsv_path, sep='\t')
    if task == 'classification':
        df['BBB+/BBB-'] = df['BBB+/BBB-'].map({'BBB-': 0, 'BBB+': 1})

    # Get all available descriptors
    calc = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    all_descriptors = calc.GetDescriptorNames()
    
    # Get indices of selected descriptors
    selected_indices = [all_descriptors.index(desc) for desc in selected_descriptors]
    
    desc_list = []
    for smiles in tqdm(df['SMILES'], 'get_molecular_descriptors'):
        mol = Chem.MolFromSmiles(smiles)
        desc = calc.CalcDescriptors(mol)
        # Only keep selected descriptors
        desc = [desc[i] for i in selected_indices]
        desc_list.append(desc)
    
    # Convert to numpy array and handle any remaining None values
    desc_array = np.array(desc_list, dtype=np.float32)
    desc_array = np.nan_to_num(desc_array)  # Replace NaN with 0
    
    # Normalize each column independently
    desc_df = pd.DataFrame(desc_array, columns=selected_descriptors)
    
    # Normalize each column
    for column in desc_df.columns:
        mean = desc_df[column].mean()
        std = desc_df[column].std()
        if std != 0:
            desc_df[column] = (desc_df[column] - mean) / std
        else:
            print(f"Warning: {column} has zero standard deviation")
            desc_df[column] = desc_df[column] - mean  # Just center if std is 0
    
    # Convert to torch tensor
    desc_tensor = torch.tensor(desc_df.values, dtype=torch.float32)
    
    return desc_tensor