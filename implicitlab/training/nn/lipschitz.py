import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deel import torchlip

from .lip_activations import Householder

#######################################################################################################################################

def DenseLipBjorck(
    dim_in: float,
    dim_hidden: float,
    n_layers: int,
    group_sort_size: int = 2, 
    bias: bool = True,
    k_coeff_lip: float=1.
):
    """
    A Lipschitz neural architecture based on Björck's orthonormalization. The implementation is based on the `SpectralLinear` layer of the deel-torchlip library.

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        group_sort_size (int, optional): Size of the GroupSort activation function. If set to zero, the activation will be a FullSort. Defaults to 2 (minimum).
        bias (bool, optional): whether to include bias vectors in the layers. Defaults to True.
        k_coeff_lip (float, optional): Lipschitz constant of the network. Defaults to 1.

    References:
        - _Sorting out Lipschitz function approximation_, Anil et al., 2019
        - [https://github.com/deel-ai/deel-torchlip](https://github.com/deel-ai/deel-torchlip)
    """
    layers = []
    activation = torchlip.FullSort if group_sort_size == 0 else lambda : torchlip.GroupSort(group_sort_size)
    layers.append(torchlip.SpectralLinear(dim_in, dim_hidden, bias=bias))
    layers.append(activation())
    for _ in range(n_layers-1):
        layers.append(torchlip.SpectralLinear(dim_hidden, dim_hidden, bias=bias))
        layers.append(activation())
    layers.append(torchlip.FrobeniusLinear(dim_hidden, 1, bias=bias))
    model = torchlip.Sequential(*layers, k_coef_lip=k_coeff_lip)
    return model

#######################################################################################################################################

def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv

class SDPBasedLipschitzDenseLayer(nn.Module):

    def __init__(self, in_features, inner_dim=-1, activation = nn.ReLU()):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = activation

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out


def DenseLipSDP(dim_in: int, dim_hidden: int, n_layers: int, coeff_lip:float = 1., activation:nn.Module = nn.ReLU(), with_group_sort: bool = True):
    """
    Neural network made of _Semi-Definite Programming_ neural layers, as proposed by [1].
    Using a square matrix $W \\in \\mathbb{R}^{k \\times k}$, a bias vector $b \\in \\mathbb{R}^k$ and an additional vector $q \\in \\mathbb{R}^k$ as parameters, each layer is defined as:

    $$x \\mapsto x - 2WT^{-1} \\sigma(W^Tx + b)$$

    where $T$ is a diagonal matrix of size $\\mathbb{R}^{k \\times k}$:

    $$ T_{ii} = \\sum_{j=1}^k \\left| (W^T W)_{ij} \\,\\exp(q_j - q_i) \\right|$$

    and $\\sigma(x) = \\max(0, x)$ is the rectified linear unit (ReLU) function. 

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        coeff_lip (float, optional): Lipschitz constant. Defaults to 1.
        activation (nn.Module, optional): Activation function. This architecture is proven to be 1-Lipschitz for ReLU, sigmoid and tanh. Defaults to ReLU.
        with_group_sort (bool, optional): whether to add a GroupSort2 between each layer to slightly increase accuracy. Defaults to True

    References:
        [1] _A Unified Algebraic Perspective on Lipschitz Neural Networks_, Araujo et al., 2023  
        [2] https://github.com/deel-ai/orthogonium/blob/main/orthogonium/layers/conv/SLL/sll_layer.py
    """
    layers = []
    layers.append(nn.ZeroPad1d((0, dim_hidden-dim_in)))
    # layers.append(torchlip.FrobeniusLinear(dim_in, dim_hidden, disjoint_neurons=False))
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDenseLayer(dim_hidden, activation=activation))
        if with_group_sort: layers.append(torchlip.GroupSort2())
    layers.append(torchlip.FrobeniusLinear(dim_hidden,1, k_coef_lip=coeff_lip))
    model = torch.nn.Sequential(*layers)
    return model

#######################################################################################################################################


class AOLLipschitzDenseLayer(nn.Module):

    def __init__(self, in_features, inner_dim=-1, activation = nn.ReLU()):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = activation

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def t(self):
        _t = torch.abs(torch.einsum('ki,kj -> ij', self.weight, self.weight)).sum(1)
        _t = safe_inv(_t)
        return torch.sqrt(_t)
    
    def forward(self, x):
        res = F.linear(self.t()*x, self.weight, self.bias)
        return self.activation(res)


def DenseLipAOL(dim_in: int, dim_hidden: int, n_layers: int, coeff_lip:float = 1., activation=nn.ReLU()):
    """
    Neural network made of _Semi-Definite Programming_ neural layers, as proposed by [1].
    Using a square matrix $W \\in \\mathbb{R}^{k \\times k}$ and a bias vector $b \\in \\mathbb{R}^k$, each layer is defined as:

    $$x \\mapsto \\sigma(WT^{-1/2}x + b)$$

    where $T$ is a diagonal matrix of size $\\mathbb{R}^{k \\times k}$:

    $$ T_{ii} = \\sum_{j=1}^k \\left| (W^T W)_{ij}\\right|$$

    and $\\sigma(x) = \\max(0, x)$ is any 1-Lipschitz function.

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        coeff_lip (float, optional): Lipschitz constant. Defaults to 1.
        activation (torch.nn.Module, optional): Activation function to consider. Defaults to nn.ReLU().

    References:
        _Almost-Orthogonal Layers for Efficient General-Purpose Lipschitz Networks_, Prach and Lampert, 2022
    """
    layers = []
    layers.append(nn.ZeroPad1d((0, dim_hidden-dim_in)))
    for _ in range(n_layers):
        layers.append(AOLLipschitzDenseLayer(dim_hidden, activation=activation))
    layers.append(torchlip.FrobeniusLinear(dim_hidden,1, k_coef_lip=coeff_lip))
    model = torch.nn.Sequential(*layers)
    return model


#######################################################################################################################################


class CPLLipschitzDenseLayer(nn.Module):

    def __init__(self, in_features, inner_dim=-1, activation = nn.ReLU(), power_it_max_iter: int = 10):
        super().__init__()
        
        self.in_features = in_features
        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = activation

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.power_it_max_iter = power_it_max_iter
        self.u = nn.Parameter(self.normalize(torch.randn(self.in_features)))
        self.u.requires_grad = False

    def normalize(self, x):
        return x / (torch.norm(x) + 1e-12)

    def l2norm_power_iteration(self, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        for _ in range(max_iter):
            v = self.normalize(F.linear(self.u, self.weight))
            self.u.data = self.normalize(F.linear(v, self.weight.T))
        z = F.linear(self.u, self.weight)
        sigma = torch.mul(z, v).sum()
        return sigma
    
    def forward(self, x):
        res = F.linear(x, self.weight, self.bias)
        res = self.activation(res) 
        res = F.linear(res, self.weight.t())
        
        sv_max = self.l2norm_power_iteration(self.power_it_max_iter)
        return x - 2/(sv_max**2 + 1e-12) * res



def DenseLipCPL(dim_in: int, dim_hidden: int, n_layers: int,  activation=nn.ReLU(), with_group_sort:bool = True):
    """
    Neural network made of _Convex Potential layers_, as proposed by [1].
    Using a square matrix $W \\in \\mathbb{R}^{k \\times k}$ and a bias vector $b \\in \\mathbb{R}^k$, each layer is defined as:

    $$x \\mapsto x - \\frac{2}{||W||_2} W^T \\sigma(Wx + b)$$

    where $\\sigma(x) = \\max(0, x)$.

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        activation (torch.nn.Module, optional): Activation function to consider. Defaults to nn.ReLU().
        with_group_sort (bool, optional): whether to add a GroupSort2 between each layer to slightly increase accuracy. Defaults to True

    References:
        _A Dynamical System Perspective for Lipschitz Neural Networks_, Meunier et al., 2022
    """
    layers = []
    layers.append(nn.ZeroPad1d((0, dim_hidden-dim_in)))
    for _ in range(n_layers):
        layers.append(CPLLipschitzDenseLayer(dim_hidden, activation=activation))
        if with_group_sort: layers.append(torchlip.GroupSort2())
    layers.append(torchlip.FrobeniusLinear(dim_hidden,1))
    model = torch.nn.Sequential(*layers)
    return model


#######################################################################################################################################


def parameter_singular_values(model):
    layers = list(model.children())
    data= []
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.linalg.svd(w)
            # data.append(f"{layer}, {s}")
            data.append(f"{layer}, min={s.min()}, max={s.max()}")
    return data