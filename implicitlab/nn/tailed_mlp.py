import torch
from torch import nn
from .siren import SirenLayer

class TailedMultiLayerPerceptron(nn.Module):
    """
    Tailed Multilayer Perceptron architecture. A MLP where the output is a sum of all layer activations up to some specified depth. Acts as a natural LoD.

    References:
        [1] T-MLP: Tailed Multi-Layer Perceptron for Level-of-Detail Signal Representation, Yang et al., 2025  
        [2] SAND: Spatially Adaptive Network Depth for Fast Sampling of Neural Implicit Surfaces, Yang et al., 2026  
    """
    def __init__(self, dim_in : int, dim_hidden : int, n_layers : int):
        """
        Args:
            dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
            dim_hidden (int): dimension of the hidden layers.
            n_layers (int): number of hidden layers.
            activ (torch.nn.Module, optional): Activation function of the network. Defaults to `torch.nn.ReLU`.
        """
        super().__init__()
        self.n_layers = n_layers
        self._depth = n_layers

        self.layers = nn.ModuleList([])
        self.layers.append(SirenLayer(dim_in, dim_hidden, is_first_layer=True))
        for _ in range(n_layers-1):
            self.layers.append(SirenLayer(dim_hidden, dim_hidden))

        self.projectors = nn.ModuleList([])
        for _ in range(n_layers):
            # Two sets of projectors multiplied
            self.projectors.append(nn.Linear(dim_hidden, 1))
            self.projectors.append(nn.Linear(dim_hidden, 1))

    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, n):
        self._depth = min(max(n, 1), self.n_layers)

    def forward(self, x, depth=None):
        depth = depth if depth is not None else self.depth
        out = 0.
        for i in range(min(self.n_layers, depth)):
            x = self.layers[i](x)
            out += self.projectors[2*i](x)*self.projectors[2*i+1](x)
        return out

    def forward_all(self, x):
        out = torch.zeros((x.shape[0], self.n_layers, 1), device=x.device)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            out[:,i] = self.projectors[2*i](x)*self.projectors[2*i+1](x)
        return torch.cumsum(out,dim=1)
