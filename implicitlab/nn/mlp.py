import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .siren import SirenLayer

class LowRankLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank):
        super().__init__()
        self.l1 = nn.Linear(dim_in,rank, bias=False)
        self.l2 = nn.Linear(rank, dim_out)

    def forward(self, x):
        return self.l2(self.l1(x))

def MultiLayerPerceptron(dim_in: int, dim_hidden: int, n_layers: int, dim_out: int = 1, activ=nn.ReLU, final_activ=nn.Identity, geometric_init:bool = False):
    """A simple customizable Multi-layer Perceptron (MLP)

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        dim_out (int, optional): dimension of the output layers. Defaults to 1.
        activ (torch.nn.Module, optional): Activation function of the network. Defaults to `torch.nn.ReLU`.
        final_activ (torch.nn.Module, optional): Final activation of the network, to be applied to the output before returning the value. Defaults to `torch.nn.Identity`.
        geometric_init (bool, optional): whether to initialize the parameter following the SAL method. Defaults to False.
    
    References:
        - _On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes_, Davies et al., 2021
        - _SAL: Sign Agnostic Learning of Shapes from Raw Data_, Atzman and Lipman, 2020
    """
    layers = []
    layers.append(nn.Linear(dim_in, dim_hidden))
    layers.append(activ())

    for _ in range(n_layers-1):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(activ())
    
    layers.append(nn.Linear(dim_hidden, dim_out))
    layers.append(final_activ())

    model = nn.Sequential(*layers)

    # if geometric_init:
    #     model.apply(geometric_init_relu_network)

    model.meta = [dim_in, dim_hidden, n_layers, activ, final_activ]
    model.id = "MLP"
    return model


def geometric_init_relu_network(model, radius_init=1.):
    n_layers = len(model)
    with torch.no_grad():
        for i in range(n_layers-1):
            layer = model[i]
            dim_out = layer.weight.size(0)
            layer.weight.normal_(mean=0.0, std=np.sqrt(2/ dim_out))
            layer.bias.data = torch.zeros_like(layer.bias.data)
    # last layer is initialized differently
    layer = model[-1]
    dim_in = layer.weight.size(-1)
    layer.weight.normal_(mean=np.sqrt(np.pi / dim_in), std=1e-6)
    layer.bias.data = torch.Tensor([-radius_init])



class MultiLayerPerceptronSkips(nn.Module):

    def __init__(self, dim_in: int, dim_hidden: int, n_layers: int, skips: list = []):
        super().__init__()
        self.skips = [x in skips for x in range(n_layers)]
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_in, dim_hidden))
        for i in range(1,n_layers-1):
            if self.skips[i+1]:
                self.layers.append(nn.Linear(dim_hidden, dim_hidden-dim_in))
            else:
                self.layers.append(nn.Linear(dim_hidden,dim_hidden))
        self.last_layer = nn.Linear(dim_hidden,1)
        self.id = "MLPS"
        self.meta = [dim_in, dim_hidden, n_layers, skips]

    def forward(self,x):
        x0 = x
        for k,layer in enumerate(self.layers):
            if self.skips[k]:
                x = layer(torch.concat((x,x0), dim=-1))
            else:
                x = layer(x)
            x = F.relu(x)
        x = self.last_layer(x)
        return x


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
