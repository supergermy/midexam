import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.PropertyRegressors import PropertyRegressors

from tqdm import tqdm

def train():
    # Hyperparameters
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    hidden_dim = 256
    output_dim = 1  # or whatever your target dimension is
    num_samples = 1000  # number of training samples
    
    # Create random training data
    rdkit_data = torch.randn(num_samples, 210)
    morgan_data = torch.randn(num_samples, 2048)
    mol2vec_data = torch.randn(num_samples, 300)
    chembert2a_data = torch.randn(num_samples, 600)
    molformer_data = torch.randn(num_samples, 768)
    
    # Create random targets
    targets = torch.randn(num_samples, output_dim)
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        rdkit_data, 
        morgan_data, 
        mol2vec_data, 
        chembert2a_data, 
        molformer_data, 
        targets
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = PropertyRegressors(hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (rdkit, morgan, mol2vec, chembert2a, molformer, target) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            
            # Forward pass
            predictions, lambdas = model(rdkit, morgan, mol2vec, chembert2a, molformer)
            
            # Calculate loss
            loss = criterion(predictions, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print batch progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Lambda weights: {(lambdas / lambdas.sum()).squeeze().detach().numpy()}')
        
        # Print epoch results
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}')

if __name__ == "__main__":
    train()