import numpy as np
import torch
import torch.nn as nn

class QuadraticLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, is_first_layer:bool=False):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_out)
        self.lin2 = nn.Linear(dim_in, dim_out)
        self.lin3 = nn.Linear(dim_in, dim_out)

        # initialization of the weights is different for the first layer
        # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
        w_std = (1 / dim_in) if is_first_layer else (np.sqrt(6. / dim_in) / 30.)
        nn.init.uniform_(self.lin1.weight, -w_std, w_std)
        nn.init.normal_(self.lin3.weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.lin2.weight, mean=0.0, std=1e-5)
        nn.init.ones_(self.lin1.bias)
        nn.init.ones_(self.lin2.bias)
        nn.init.zeros_(self.lin3.bias)

    def forward(self, x):
        return torch.mul(self.lin1(x), self.lin2(x)) + self.lin3(torch.square(x))
    

class QuaNet(nn.Module):
    def __init__(self,
        dim_in: int,
        dim_hidden: int,
        n_layers: int,
        dim_out = 1,
        activation=nn.Softplus,
        residual: bool = False
    ):
        """
        Quadratic deep network implementation.

        Code adapted from [https://github.com/Galaxeaaa/HotSpot/blob/main/models/Net.py](https://github.com/Galaxeaaa/HotSpot/blob/main/models/Net.py)

        Args:
            dim_in (int): dimension of the input vector. Usually 2 or 3 for neural implicits.
            dim_hidden (int): dimension of the hidden layers.
            n_layers (int): number of hidden layers.
            dim_out (int, optional): dimension of the output layers. Defaults to 1.
            activation (nn.Callable, optional): Activation function to be used between each layer. Defaults to nn.Softplus.
            residual (bool, optional): whether to consider residual connections between layers. Defaults to False.
        
        References:
            _Universal Approximation with Quadratic Deep Networks_, Fan et al., 2019
        """
        super().__init__()
        dims = [dim_in] + [dim_hidden for _ in range(n_layers)] + [dim_out]
        self.num_layers = len(dims)
        self.residual = residual
        
        for l in range(self.num_layers - 1):
            qua = QuadraticLayer(dims[l], dims[l + 1], is_first_layer=(l==0))
            setattr(self, "qua" + str(l), qua)
        self.activation = activation()

    def forward(self, inputs):
        x = inputs
        for l in range(self.num_layers - 1):
            qua = getattr(self, "qua" + str(l))
            if l == self.num_layers - 2:
                x = qua(x)
            else:
                if self.residual and l>0:
                    x = self.activation(qua(x)) + x
                else:
                    x = self.activation(qua(x))
        return x