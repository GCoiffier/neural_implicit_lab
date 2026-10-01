import torch
from torch import nn
import numpy as np

class SinusActivation(nn.Module):
    def __init__(self, w0 = 6.):
        super().__init__()
        self.w0 = w0
        
    def forward(self, x):
        return torch.sin(self.w0*x)   

    
class SirenLayer(nn.Module):
    def __init__(self, 
        dim_in:int, 
        dim_out:int, 
        w0 : float = 30., 
        activation:bool=True,
        is_first_layer=False,
        residual:bool=False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim_out, dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_out))

        with torch.no_grad():
            stdv = (1 / dim_in) if is_first_layer else np.sqrt(6 / dim_in) / 30.
            self.weight.uniform_(-stdv,stdv)
            self.bias.uniform_(-stdv, stdv)

        self.activation = SinusActivation(w0) if activation else nn.Identity()
        self.residual = residual if dim_in==dim_out else False

    def forward(self, x):
        out = self.activation(nn.functional.linear(x, self.weight, self.bias))
        if self.residual: return out+x
        return out


class SirenGeometricInitializer:
    """
    References:
        - _DiGS : Divergence guided shape implicit neural representation for unoriented  point clouds_, Y.Ben-Shabat et al., 2022
        - _Improving Accuracy and Efficiency of Implicit Neural Representations: Making SIREN a WINNER_, H.Chandravamsi et al., 2026
    """

    def __init__(self, init_type="geometric", w0=30.):
        self.init_type = init_type.lower()
        self.w0 = w0

    def apply(self, model):
        if self.init_type == "geometric":
            model.apply(self.geom_sine_init)
            model[-1].apply(self.geom_sine_init_last_layer)
            model[-2].apply(self.geom_sine_init_second_last_layer)

        elif self.init_type == "mfgi":
            self.periods = [1,30]  # Number of periods of sine the values of each section of the output vector should hit
            self.portion_per_period = np.array([0.25, 0.75])  # Portion of values per section/period
            model.apply(self.geom_sine_init)
            model[0].apply(self.mfgi_init_first_layer)
            model[1].apply(self.mfgi_init_second_layer)
            model[-2].apply(self.geom_sine_init_second_last_layer)
            model[-1].apply(self.geom_sine_init_last_layer)

        elif self.init_type == "winner":
            self.s = 1.
            model[0].apply(self.winner_init)
            model[1].apply(self.winner_init)

    ####### geometric initialization #######

    def geom_sine_init(self, m):
        if not hasattr(m, "weight"): return
        with torch.no_grad():
            dim_out = m.weight.size(0)
            w_bnd = np.sqrt(3 / dim_out)/self.w0
            m.weight.uniform_(-w_bnd, w_bnd)
            b_bnd = 1 / (dim_out * 1000 * self.w0)
            m.bias.uniform_(-b_bnd, b_bnd)


    def geom_sine_init_second_last_layer(self, m):
        if not hasattr(m, "weight"): return
        with torch.no_grad():
            dim_out = m.weight.size(0)
            assert m.weight.shape == (dim_out, dim_out)
            m.weight.data = np.pi/2 * torch.eye(dim_out) \
                        + 0.001 * torch.randn(dim_out, dim_out)
            m.bias.data = np.pi/2 * torch.ones(dim_out,) \
                        + 0.001 * torch.randn(dim_out)
            m.weight.data /= self.w0
            m.bias.data /= self.w0

    def geom_sine_init_last_layer(self, m):
        if not hasattr(m, "weight"): return
        with torch.no_grad():
            dim_in = m.weight.size(-1)
            assert m.weight.shape == (1, dim_in)
            assert m.bias.shape == (1,)
            m.weight.data = torch.full((1, dim_in), fill_value=-1.) + 0.001 * torch.randn(dim_in)
            m.bias.data = torch.zeros(1) + dim_in-1

    ####### multi frequency geometric initialization #######

    def mfgi_init_first_layer(self, m):
        if not hasattr(m, "weight"): return
        with torch.no_grad():
            dim_in = m.weight.size(-1)
            dim_out = m.weight.size(0)
            dim_per_period = (self.portion_per_period * dim_out).astype(int)  # Number of values per section/period
            weights = []
            for i in range(0, len(self.periods)):
                period = self.periods[i]
                dim_period = dim_per_period[i]
                scale = self.w0 / period
                weight = torch.zeros(dim_period, dim_in)
                weight.uniform_(-np.sqrt(3 / dim_in) / scale, np.sqrt(3 / dim_in) / scale)
                weights.append(weight)
            W0_new = torch.cat(weights, axis=0)
            m.weight.data = W0_new

    def mfgi_init_second_layer(self, m):
        if not hasattr(m, "weight"): return
        with torch.no_grad():
            dim_in = m.weight.size(-1)
            dim_per_period = (self.portion_per_period * dim_in).astype(int) # Number of values per section/period
            k = dim_per_period[0]  # the portion that only hits the first period
            bnd = np.sqrt(3 / dim_in) / self.w0
            W1_new = torch.zeros(dim_in, dim_in).uniform_(-bnd,bnd)*5e-4
            W1_new_1 = torch.zeros(k, k).uniform_(-bnd, bnd)
            W1_new[:k, :k] = W1_new_1
            m.weight.data = W1_new

    ####### Winner  initialization #######
    def winner_init(self, m):
        if not hasattr(m, "weight"): return
        m.weight.data += torch.randn_like(m.weight.data)*self.s/self.w0


def SirenNet(
    dim_in: int, 
    dim_hidden: int, 
    n_layers: int, 
    dim_out: int =1, 
    w0: float = 6.,
    w0_first_layer : float = 30.,
    residual: bool = False,
    init_type="siren"
    ):
    """
    
    Code adapted from [https://github.com/MClemot/SkeletonLearning/blob/main/siren_nn.py](https://github.com/MClemot/SkeletonLearning/blob/main/siren_nn.py)

    Args:
        dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
        dim_hidden (int): dimension of the hidden layers.
        n_layers (int): number of hidden layers.
        dim_out (int, optional): dimension of the output layers. Defaults to 1.
        w0 (float, optional): Frequency of the activation function. Defaults to 6.
        w0_first_layer (float, optional): Frequency of the activation function in the first layer. Defaults to 30.
        residual (bool, optional): Whether to include residual connections between each layer. Defaults to False.
    
    References:
        _Implicit Neural Representations with Periodic Activation Functions_, Sitzmann et al., 2020
    """
    layers = []
    # First dim_in -> dim_hidden layer
    layers.append(SirenLayer(dim_in, dim_hidden, w0_first_layer, is_first_layer=True))
    # Intermediate dim_hidden -> dim_hidden layers
    for _ in range(n_layers-1):
        layers.append(SirenLayer(dim_hidden, dim_hidden, w0, residual=residual))
    # last dim_hidden->1 layer has no activation
    layers.append(SirenLayer(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, activation = False))
    model = nn.Sequential(*layers)

    # Apply initialization
    if init_type.lower() != "siren":
        initializer = SirenGeometricInitializer(init_type.lower())
        initializer.apply(model)

    model.id = "SIREN"
    model.meta = [dim_in, dim_hidden, n_layers]
    return model



class QuadraticSkipSirenNet(nn.Module):
    """
    SIREN with explicit quadratic skip:
        f(x) = f_siren(x) + x^T A(x) x
    
    References:
        https://github.com/sweidemaier/Neat_SDF/blob/master/models/QuadNet.py

    """

    def __init__(self, 
        dim_in: int, 
        dim_hidden: int, 
        n_layers: int, 
        dim_out: int = 1, 
        w0: float = 6., 
        w0_first_layer: float = 30., 
        residual: bool = False
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.siren = SirenNet(dim_in, dim_hidden, n_layers-1, dim_out=dim_hidden, w0=w0, w0_first_layer=w0_first_layer, residual=residual)
        self.last_siren = SirenLayer(dim_hidden, dim_out, w0=w0, activation=False)
    
        self.quad_head = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_hidden),
            SinusActivation(w0=30.),
            nn.Linear(self.dim_hidden, self.dim_in * self.dim_in)
        )

        with torch.no_grad():
            w_std = np.sqrt(6 / dim_in) / 30.
            self.quad_head[0].weight.uniform_(-w_std, w_std)
            self.quad_head[2].weight.uniform_(-w_std, w_std)        

    def forward(self, x):
        net = x
        net = self.siren(net)
        siren_out = self.last_siren(net)
        A = self.quad_head(net).view(
            net.shape[0], self.dim_in, self.dim_in
        )
        quad = torch.einsum('...i,...ij,...j->...', x, A, x)
        quad = quad.unsqueeze(-1)
        return siren_out + quad