# With help from:
#   https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

import torch
from sklearn.model_selection import KFold
import torch
import ray
from ray import tune
import os

from tqdm import tqdm
from data_controller import EmbeddingDataset

from model import Model

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
  k_folds = 5
  loss_function = torch.nn.functional.cosine_similarity
  
  # For fold results
  results = {}
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  dataset = config['dataset']

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=config['batch_size'], 
                      sampler=train_subsampler
                    )
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=config['batch_size'], 
                      sampler=test_subsampler
                    )
    
    # Init the neural network
    network = config['model'](config['input_shape'], config['output_shape'])

    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    # network.to(device)

    network.apply(reset_weights)

    # Initialize optimizer
    optimizer = config['optimizer'](network.parameters(), lr=config['learning_rate'])
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, config['epochs']):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}"), 0):        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets, dim=1)

        loss = 1 - loss.mean()
        
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
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):

        # Get inputs
        inputs, targets = data

        # Generate outputs
        outputs = network(inputs)

      # Calculate the loss
      cos_sim = torch.nn.functional.cosine_similarity(outputs, targets, dim=1)
      loss = 1 - cos_sim.mean()

      # Print Loss for fold
      print(f"Loss of fold {fold}: {loss}")
      print('--------------------------------')
      results[fold] = loss

  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  average_loss = sum/len(results.items())
  print(f"Average loss: {average_loss}")
  

  tune.report({"average_loss": average_loss.item()})

# Dummy dataset for testings
# class DummyDataset(Dataset):
#     def __init__(self, in_shape, out_shape, num_samples=1000):
#         self.X = torch.randn(num_samples, in_shape)
#         self.y = torch.randn(num_samples, out_shape)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

if __name__ == '__main__':
  
  # Create dumby dataset

  # dataset = DummyDataset(in_shape=input_shape, out_shape=output_shape, num_samples=1000)

  image_dir = os.path.abspath("../data/formatted_data/train/image_embeddings")
  text_dir = os.path.abspath("../data/formatted_data/train/text_embeddings")

  dataset = EmbeddingDataset(image_dir, text_dir)

  input_shape, output_shape = dataset.get_feature_sizes()

  search_space = {
    'dataset': dataset,
    'model': Model,
    'input_shape': input_shape,
    'output_shape': output_shape,
    'learning_rate': tune.loguniform(1e-5, 1e-1),
    'epochs': tune.choice([5]),
    'optimizer': tune.choice([torch.optim.Adam]),
    'batch_size': tune.choice([2, 4, 8, 16, 32])
  }

  ray.init(ignore_reinit_error=True)
  analysis = tune.run(
    trainable,
    config=search_space,
    verbose=1
  )
