import torch
import torch.nn as nn

class Model(nn.Module):
    '''
    Customizable Feed-Forward Neural Network
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

# Example usage:
# model = FeedForwardNN(input_size=100, output_size=10)
# x = torch.randn(1, 100)  # Batch size of 1, input of size 100
# output = model(x)
