import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deel import torchlip

#######################################################################################################################################

def DenseLipBjorck(
    dim_in:float,
    dim_hidden:float,
    n_layers:int,
    group_sort_size:int=0, 
    k_coeff_lip:float=1., 
    n_iter_spectral:int=3,
    n_iter_bjorck:int=15
):
    """
    A Lipschitz neural architecture based on Björck's orthonormalization. The implementation is based on the `SpectralLinear` layer of the deel-torchlip library.

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        group_sort_size (int, optional): Size of the GroupSort activation function. If set to zero, the activation will be a FullSort. Defaults to 0.
        k_coeff_lip (float, optional): Lipschitz constant of the network. Defaults to 1.
        n_iter_spectral (int, optional): Number of spectral iterations for Lipschitz constant correction. Defaults to 3.
        n_iter_bjorck (int, optional): Number of Björck orthonormalisation iterations for Lipschitz constant correction. Defaults to 15.

    References:
        - _Sorting out Lipschitz function approximation_, Anil et al., 2019
        - [https://github.com/deel-ai/deel-torchlip](https://github.com/deel-ai/deel-torchlip)
    """
    layers = []
    activation = torchlip.FullSort if group_sort_size == 0 else lambda : torchlip.GroupSort(group_sort_size)
    layers.append(torchlip.SpectralLinear(dim_in, dim_hidden, niter_spectral=n_iter_spectral, niter_bjorck=n_iter_bjorck))
    layers.append(activation())
    for _ in range(n_layers-1):
        layers.append(torchlip.SpectralLinear(dim_hidden, dim_hidden, niter_spectral=n_iter_spectral, niter_bjorck=n_iter_bjorck))
        layers.append(activation())
    layers.append(torchlip.FrobeniusLinear(dim_hidden, 1))
    model = torchlip.Sequential(*layers, k_coef_lip=k_coeff_lip)
    model.meta = [dim_in, dim_hidden, n_layers]
    model.id = "Spectral"
    return model


#######################################################################################################################################

def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv

class SDPBasedLipschitzDense(nn.Module):

    def __init__(self, in_features, inner_dim=-1):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = nn.ReLU()

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
    
def DenseLipSDP(dim_in: int, dim_hidden: int, n_layers: int, coeff_lip:float = 1.):
    """
    Neural network made of _Semi-Definite Programming_ neural layers, as proposed by [1].
    Using a square matrix $W \in \mathbb{R}^{k \\times k}$, a bias vector $b \in \mathbb{R}^k$ and an additional vector $q \in \mathbb{R}^k$ as parameters, each layer is defined as:

    $$x \mapsto x - 2WT^{-1} \sigma(W^Tx + b)$$

    where $T$ is a diagonal matrix of size $\mathbb{R}^{k \\times k}$:

    $$ T_{ii} = \sum_{j=1}^k \left| (W^T W)_{ij} \,\exp(q_j - q_i) \\right|$$

    and $\sigma(x) = \max(0, x)$ is the rectified linear unit (ReLU) function. 

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        coeff_lip (float, optional): Lipschitz constant. Defaults to 1.

    References:
        [1] _A Unified Algebraic Perspective on Lipschitz Neural Networks_, Araujo et al., 2023
    """
    layers = []
    layers.append(nn.ZeroPad1d((0, dim_hidden-dim_in)))
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(dim_hidden))
    layers.append(torchlip.FrobeniusLinear(dim_hidden,1, k_coef_lip=coeff_lip))
    model = torch.nn.Sequential(*layers)
    model.id = "SDP"
    model.meta = [dim_in, dim_hidden, n_layers]
    return model

#######################################################################################################################################
