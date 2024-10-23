import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.PropertyRegressors import PropertyRegressors
from datasets.get_fingerprints import get_fingerprints
from datasets.get_molecular_descriptors import get_molecular_descriptors
from datasets.get_chembert2as import get_chembert2as
from datasets.get_molformers import get_molformers

from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train(rdkit_data, morgan_data, chembert2a_data, molformer_data, label_tensor):
    # Hyperparameters
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    hidden_dim = 256
    output_dim = 1  # or whatever your target dimension is
    
    # # Create random training data
    # rdkit_data = torch.randn(num_samples, 210)
    # morgan_data = torch.randn(num_samples, 2048)
    # chembert2a_data = torch.randn(num_samples, 600)
    # molformer_data = torch.randn(num_samples, 768)
    
    # Create random targets
    targets = torch.tensor(label_tensor, device=device)

    for data in rdkit_data, morgan_data, chembert2a_data, molformer_data:
        assert not torch.isnan(data).any()
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        rdkit_data, 
        morgan_data, 
        chembert2a_data, 
        molformer_data, 
        targets
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = PropertyRegressors(hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    epoch_loss = 0.0
    
    for epoch in range(epochs):
        
        for batch_idx, (rdkit, morgan, chembert2a, molformer, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            predictions, lambdas = model(rdkit, morgan, chembert2a, molformer)
            
            # Calculate loss
            loss = criterion(predictions, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # # Print batch progress
            # if batch_idx % 10 == 0:
            #     print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            #     print(f'Lambda weights: {(lambdas / lambdas.sum()).squeeze().detach().numpy()}')
        
        # Print epoch results
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}')
        print(lambdas)
        epoch_loss = 0.0

if __name__ == "__main__":

    descriptors = get_molecular_descriptors(task='classification').to(device)
    print(f"descriptor tensor shape: {descriptors.shape}")  # [num_samples, 210]
    
    fingerprints, label_tensor = get_fingerprints(task='classification')
    print(f"fingerprint tensor shape: {fingerprints.shape}")  # [num_samples, 2048]

    chembert2as = get_chembert2as(task='classification').to(device)
    print(f"chembert2a tensor shape: {chembert2as.shape}")  # [num_samples, 600]

    molformers = get_molformers(task='classification').to(device)
    print(f"molformers tensor shape: {molformers.shape}")  # [num_samples, 768]

    train(descriptors, fingerprints, chembert2as, molformers, label_tensor)