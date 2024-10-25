import torch
from torch.utils.data import Dataset, DataLoader

class MolecularDataset(Dataset):
    def __init__(self, descriptors, fingerprints, chembert2as, molformers, labels):
        """
        Initialize the dataset with the different molecular representations
        
        Args:
            descriptors (torch.Tensor): Shape [7807, 210]
            fingerprints (torch.Tensor): Shape [7807, 2048]
            chembert2as (torch.Tensor): Shape [7807, 600]
            molformers (torch.Tensor): Shape [7807, 768]
            labels (torch.Tensor): Shape [7807, 1]
        """
        self.descriptors = descriptors
        self.fingerprints = fingerprints
        self.chembert2as = chembert2as
        self.molformers = molformers
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Return a single sample from the dataset
        
        Args:
            idx (int): Index of the sample to return
            
        Returns:
            dict: Dictionary containing all features and label for the sample
        """
        return {
            'descriptors': self.descriptors[idx],
            'fingerprints': self.fingerprints[idx],
            'chembert2as': self.chembert2as[idx],
            'molformers': self.molformers[idx],
            'label': self.labels[idx],
        }

# Example usage:
def create_data_loaders(descriptors, fingerprints, chembert2as, molformers, labels, 
                       batch_size=32, train_split=0.8, random_seed=42):
    """
    Create train and validation data loaders
    
    Args:
        descriptors (torch.Tensor): Molecular descriptors
        fingerprints (torch.Tensor): Molecular fingerprints
        chembert2as (torch.Tensor): ChemBERT embeddings
        molformers (torch.Tensor): Molformer embeddings
        labels (torch.Tensor): Target labels
        batch_size (int): Batch size for the data loaders
        train_split (float): Proportion of data to use for training
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Calculate split indices
    dataset_size = len(labels)
    indices = torch.randperm(dataset_size)
    train_size = int(train_split * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train dataset
    train_dataset = MolecularDataset(
        descriptors[train_indices],
        fingerprints[train_indices],
        chembert2as[train_indices],
        molformers[train_indices],
        labels[train_indices]
    )
    
    # Create validation dataset
    val_dataset = MolecularDataset(
        descriptors[val_indices],
        fingerprints[val_indices],
        chembert2as[val_indices],
        molformers[val_indices],
        labels[val_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader
