import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .implicit_model import ImplicitModel


class LowRankLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank):
        super().__init__()
        self.l1 = nn.Linear(dim_in,rank, bias=False)
        self.l2 = nn.Linear(rank, dim_out)

    def forward(self, x):
        return self.l2(self.l1(x))


class MultiLayerPerceptron(ImplicitModel):

    def __init__(self, dim_in: int, dim_hidden: int, n_layers: int, dim_out: int = 1, activ=nn.ReLU, final_activ=nn.Identity, geometric_init:bool = False):
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
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim_in, dim_hidden))
        layers.append(activ())

        for _ in range(n_layers-1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activ())
        
        layers.append(nn.Linear(dim_hidden, dim_out))
        layers.append(final_activ())

        self.model = nn.Sequential(*layers)
        if geometric_init:
            self._initialize_geometric_relu_network()

    def forward(self, x):
        return self.model(x)


    def _initialize_geometric_relu_network(self, radius_init=1.):
        n_layers = len(self.model)
        with torch.no_grad():
            for i in range(n_layers-1):
                layer = self.model[i]
                dim_out = layer.weight.size(0)
                layer.weight.normal_(mean=0.0, std=np.sqrt(2/ dim_out))
                layer.bias.data = torch.zeros_like(layer.bias.data)
        # last layer is initialized differently
        layer = self.model[-1]
        dim_in = layer.weight.size(-1)
        layer.weight.normal_(mean=np.sqrt(np.pi / dim_in), std=1e-6)
        layer.bias.data = torch.Tensor([-radius_init])



class MultiLayerPerceptronSkips(ImplicitModel):

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