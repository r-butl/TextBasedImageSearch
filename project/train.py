# With help from:
#   https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

import torch
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
import ray
from ray import tune

from model import Model

global input_shape
global output_shape

input_shape = 512
output_shape= 512

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def trainable(config):

    # Configuration options
    loss_function = torch.nn.functional.cosine_similarity

    # Set fixed random number seed
    torch.manual_seed(42)

    dataset = config['dataset']

    dataset_size = len(dataset)
    val_split = int(0.2 * dataset_size)
    train_split = dataset_size - val_split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'])
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'])

    # Init the neural network
    network = config['model'](input_shape, output_shape)

    network.apply(reset_weights)

    # Initialize optimizer
    optimizer = config['optimizer'](network.parameters(), lr=config['learning_rate'])

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    # Run the training loop for defined number of epochs
    for epoch in range(0, config['epochs']):

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = network(inputs)

            # Compute loss
            loss = loss_function(outputs, targets, dim=1)

            loss = 1 - loss.mean()  # for cosine similarity loss

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0

        # Validation after epoch
        network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in valloader:
                val_inputs, val_targets = val_data
                val_outputs = network(val_inputs)
                val_cos_sim = torch.nn.functional.cosine_similarity(val_outputs, val_targets, dim=1)
                val_loss += (1 - val_cos_sim.mean()).item()
        val_loss /= len(valloader)
        print(f"Validation Loss after epoch {epoch+1}: {val_loss}")

        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model.")
            best_val_loss = val_loss
            torch.save(network.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break
        network.train()
        
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    tune.report({"final_val_loss": val_loss})

# Dummy dataset for testings
class DummyDataset(Dataset):
    def __init__(self, in_shape, out_shape, num_samples=1000):
        self.X = torch.randn(num_samples, in_shape)
        self.y = torch.randn(num_samples, out_shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
  
    # Create dumby dataset

    dataset = DummyDataset(in_shape=input_shape, out_shape=output_shape, num_samples=1000)

    config = {
    'dataset': dataset,
    'model': Model,
    'learning_rate': 1e-5,
    'epochs': 5,
    'optimizer': torch.optim.Adam,
    'batch_size': 16
    }

    trainable(config)