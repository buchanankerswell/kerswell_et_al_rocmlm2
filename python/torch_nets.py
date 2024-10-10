#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
import torch.nn as nn

#######################################################
## .1.               PyTorch Nets                !!! ##
#######################################################
class SimpleNet(nn.Module):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        """
        super(SimpleNet, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # Build the model layers
        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(inplace=True))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):
        """
        """
        return self.net(x)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        """
        """
        params_info = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "num_layers": len(self.hidden_layer_sizes) + 1,
            "hidden_layer_sizes": self.hidden_layer_sizes
        }

        return params_info

#######################################################
class ImprovedNet(nn.Module):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate=0.1):
        super(ImprovedNet, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):
        """
        """
        return self.net(x)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        """
        """
        params_info = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dropout rate": self.dropout_rate,
            "num_layers": len(self.hidden_layer_sizes) + 1,
            "hidden_layer_sizes": self.hidden_layer_sizes
        }

        return params_info
