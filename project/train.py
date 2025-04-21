# With help from:
#   https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

import torch
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
import ray
from ray import tune
import os

from model import Model
from data_controller import EmbeddingDataset

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

    train_dataset = config['train_dataset']
    validate_dataset = config['validate_dataset']

    # Set training and validation datasets instead 
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(validate_dataset, batch_size=config['batch_size'])

    # Init the neural network
    network = config['model'](config['input_shape'], config['output_shape'], config['layers'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    network.to(device)

    network.apply(reset_weights)

    # Initialize optimizer
    optimizer = config['optimizer'](network.parameters(), lr=config['learning_rate'])

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience = 7
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
            inputs, targets = inputs.to(device), targets.to(device)

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
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

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


if __name__ == '__main__':

    train_image_dir = os.path.abspath("../data/formatted_data/train/image_embeddings")
    train_text_dir = os.path.abspath("../data/formatted_data/train/text_embeddings")
    train_dataset = EmbeddingDataset(train_image_dir, train_text_dir)

    validate_image_dir = os.path.abspath("../data/formatted_data/validate/image_embeddings")
    validate_text_dir = os.path.abspath("../data/formatted_data/validate/text_embeddings")
    validate_dataset = EmbeddingDataset(validate_image_dir, validate_text_dir)

    input_shape, output_shape = train_dataset.get_feature_sizes()

    config = {
    'train_dataset': train_dataset,
    'validate_dataset': validate_dataset,
    'model': Model,
    'input_shape': input_shape,
    'output_shape': output_shape,
    'layers': 
      [1024, 768, 512, 384]
    ,
    'learning_rate': 1e-4,
    'epochs': 100,
    'optimizer': torch.optim.Adam,
    'batch_size': 16
    }

    trainable(config)