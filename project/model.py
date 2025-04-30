import torch
import torch.nn as nn

class Model(nn.Module):
    '''
    Customizable Feed-Forward Neural Network with BatchNorm and Dropout
    '''
    def __init__(self, input_size, output_size, layers=[], dropout=0.3):
        super().__init__()
        layer_list = []
        prev_size = input_size

        for size in layers:
            layer_list.append(nn.Linear(prev_size, size))
            layer_list.append(nn.BatchNorm1d(size))  # Normalization
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p=dropout))
            prev_size = size

        layer_list.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
