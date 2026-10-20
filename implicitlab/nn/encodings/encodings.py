import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..implicit_model import ImplicitModel


class FourierEncoding(ImplicitModel):
    """
    Positional encoding used in neural radiance fields. Maps the input $x$ to

    $$[\\sin(x),\\, \\cos(x),\\, \\sin(4x),\\, \\cos(4x),\\, ...,\\, \\sin(2^n\\,x),\\, \\cos(2^n\\,x)]$$
    """
    def __init__(self, input_dim: int, encoded_dim: int = 10):
        """
        Args:
            input_dim (int): dimension of the input points (usually 2 or 3).
            encoded_dim (int): number of frequencies (i.e. half the size of the final encoded vector). Defaults to 10
        """
        super().__init__()
        self.input_dim = input_dim
        if encoded_dim<1:
            raise ValueError("`encoded_dim` should be > 0")
        self.encoded_dim = encoded_dim
        freqs = 2. ** torch.linspace(0., encoded_dim-1)
        self.register_buffer('freqs', freqs, persistent=False)

    @property
    def output_dim(self):
        return self.input_dim * self.encoded_dim * 2
    
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=-1) * self.freqs # (..., dim, L)
        return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)


class RandomFourierEncoding(ImplicitModel):

    def __init__(self, input_dim: int, encoded_dim: int, stdv: float = 1.):
        """Encodes a given position into different frequencies of trigonometric functions :

        $$x \\mapsto [\\cos(2\\pi b x), \\sin(2 \\pi b x)]$$

        where $b$ is sampled from a normal distribution of zero mean (and provided standard deviation)

        Args:
            input_dim (int): dimension of the input points (usually 2 or 3).
            encoded_dim (int): number of frequencies (i.e. half the size of the final encoded vector. the tensor `b` will have shape `(geometry.dim, 2*encoded_dim)`
            stdv (float, optional): Standard variation of the normal distribution used to compute `b`. Defaults to 1..

        Raises:
            ValueError: Fails if the encoding size is not an even number.

        References:
            - Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains, Tancik et al. (2020) : [https://arxiv.org/abs/2006.10739](https://arxiv.org/abs/2006.10739)
            - [https://github.com/jmclong/random-fourier-features-pytorch/tree/main](https://github.com/jmclong/random-fourier-features-pytorch/tree/main)
        """
        super().__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        if encoded_dim<1:
            raise ValueError("`encoded_dim` should be > 0")
        b = np.random.normal(0, stdv, size=(input_dim, encoded_dim//2))
        self.register_buffer("b", torch.FloatTensor(b))

    @property
    def output_dim(self):
        return self.input_dim * self.encoded_dim

    def forward(self, x: torch.Tensor):
        vp = 2 * torch.pi * x @ self.b
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class HyperplaneEncoding(ImplicitModel):
    def __init__(self, points : torch.Tensor, normals : torch.Tensor, trainable:bool=False):
        """Encoding that considers points with normals sampled on the geometry and applies the signed distance to each corresponding hyperplanes:

        $$x \\mapsto \\langle n, x-p \\rangle$$

        where $(p,n)$ are points and normals.

        Args:
            points (torch.Tensor): the points to consider
            normals (torch.Tensor): the normals associated with each point. Will be normalized.
            trainable (bool, optional): whether the points and normals can be trained as model parameters. Defaults to False.
        """
        super().__init__()
        normals = torch.as_tensor(normals, dtype=torch.float32)
        normals = F.normalize(normals, dim=1) # normals should be of unit norm
        self.normals = nn.Parameter(normals, requires_grad=trainable)
        self.points = nn.Parameter(torch.as_tensor(points, dtype=torch.float32), requires_grad=trainable)
        
        # Make the save config point to the parameter Tensors and not the input, 
        # so that tensors are not duplicated between config and state_dict when saving on the disk
        self._config["points"] = self.points.detach()
        self._config["normals"] = self.normals.detach() 

    def forward(self, x: torch.Tensor):
        c = torch.unsqueeze(x, 1) - self.points
        return torch.sum(F.normalize(self.normals, dim=1) * c, dim=-1) 
    


class PointDistanceEncoding(ImplicitModel):
    def __init__(self, points: torch.Tensor, trainable:bool = False):
        """Encodes the input point x with its l2 distance to sampled points $p$ in space:

        $$x \\mapsto ||x-p||_2$$

        Args:
            points (torch.Tensor): The points consider for encoding
            trainable (bool, optional): whether the points can be trained as model parameters. Defaults to False.
        """
        super().__init__()
        self.points = nn.Parameter(torch.as_tensor(points, dtype=torch.float32), requires_grad=trainable)
        # Make the save config point to the parameter Tensors and not the input, 
        # so that tensors are not duplicated between config and state_dict when saving on the disk
        self._config["points"] = self.points.detach()

    def forward(self, x: torch.Tensor):
        c = torch.unsqueeze(x, 1) - self.points
        return torch.norm(c, dim=-1)


class GaussianPointDistanceEncoding(PointDistanceEncoding):
    def __init__(self,  points: torch.Tensor, stdv:float = 1., trainable:bool = False):
        """ Variant of the PointDistanceEncoding where distance are then fed into a Gaussian kernel:

        $$x \\mapsto e^{ - \\frac{||x-p||^2}{\\sigma}}$$

        Args:
            points (torch.Tensor): The points consider for encoding
            stdv (float): standard deviation of the Gaussian kernel. Defaults to 1.
            trainable (bool, optional): whether the points can be trained as model parameters. Defaults to False.
        """
                
        super().__init__(points, trainable)
        self.stdv = stdv

    def forward(self, x: torch.Tensor):
        c = torch.unsqueeze(x, 1) - self.points
        norm2 = torch.sum(c*c, dim=-1)
        return torch.exp( -norm2/self.stdv)