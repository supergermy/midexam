import pandas as pd
import numpy as np
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
import torch
from tqdm import tqdm



def get_mol2vec(task):
    """Convert SMILES to molecular descriptors and return as torch tensor"""
    if task == 'classification':
        tsv_path = 'B3DB_classification.tsv'
    elif task == 'regression':
        tsv_path = 'B3DB_regression.tsv'
    else:
        raise KeyError(f'task should be either "classification" or "regression", but got {task}')
    df = pd.read_csv(tsv_path, sep='\t')
    if task == 'classification':
        df['BBB+/BBB-'] = df['BBB+/BBB-'].map({'BBB-': 0, 'BBB+': 1})
    
    model = word2vec.Word2Vec.load('datasets/model_300dim.pkl')
    m2v_list = []

    for smiles in tqdm(df['SMILES'], 'MolFromSmiles'):
        mol = Chem.MolFromSmiles(smiles)
        m2v =  DfVec(sentences2vec(MolSentence(mol2alt_sentence(mol, radius=2)), model, unseen='UNK'))
        m2v_list.append(m2v)
    
    # Convert to numpy array and handle any remaining None values
    m2v_array = np.array(m2v_list, dtype=np.float32)
    m2v_array = np.nan_to_num(m2v_array)  # Replace NaN with 0
    
    # Convert to torch tensor
    m2v_array = torch.tensor(m2v_array, dtype=torch.float32)
    
    return m2v_array
