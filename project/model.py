import torch
import torch.nn as nn

class Model(nn.Module):
    '''
    Customizable Feed-Forward Neural Network
    '''
    def __init__(self, input_size, output_size, layers=[]):
        super().__init__()
        layer_list = []
        prev_size = input_size

        for size in layers:
            layer_list.append(nn.Linear(prev_size, size))
            layer_list.append(nn.ReLU())
            prev_size = size

        layer_list.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

