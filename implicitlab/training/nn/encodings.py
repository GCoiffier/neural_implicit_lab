import torch
from torch import nn
import torch.nn.functional as F
import mouette as M
import numpy as np

from ...data.sample_utils import sample_points_and_normals2D

class RandomFourierEncoding(nn.Module):

    def __init__(self, geometry : M.mesh.Mesh, dim_encoded: int, stdv: float = 1.):
        """Encodes a given position into different frequencies of trigonometric functions :

        $$x \mapsto [\cos(2\pi b x), \sin(2 \pi b x)]$$

        where $b$ is sampled from a normal distribution of zero mean (and provided standard deviation)

        Args:
            geometry (mouette.Mesh): the input geometry. Only the dimensionality attribute (`geometry.dim`) is accessed, but argument is kept for consistency with other encodings.
            dim_encoded (int): size of the final encoded vector. Should be an even number (as both sin and cos of every frequency is computed). the tensor `b` will have shape `(geometry.dim, dim_encoded//2)`
            stdv (float, optional): Standard variation of the normal distribution used to compute `b`. Defaults to 1..

        Raises:
            ValueError: Fails if the encoding size is not an even number.

        References:
            - Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains, Tancik et al. (2020) : [https://arxiv.org/abs/2006.10739](https://arxiv.org/abs/2006.10739)
            - [https://github.com/jmclong/random-fourier-features-pytorch/tree/main](https://github.com/jmclong/random-fourier-features-pytorch/tree/main)
        """
        super().__init__()
        dim_in = geometry.dim
        if dim_encoded%2==1:
            raise ValueError("The size should be an even number.")
        b = np.random.normal(0, stdv, size=(dim_in, dim_encoded//2))
        self.b = nn.Parameter(torch.FloatTensor(b), requires_grad=False)

    def forward(self, x):
        vp = 2 * torch.pi * x @ self.b
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class HalfPlaneEncoding(nn.Module):
    def __init__(self, geometry, n_points: int):
        """Encoding that considers points with normals sampled on the geometry and applies the signed distance to each corresponding hyperplanes:

        $$x \mapsto \langle n, x-p \\rangle$$

        where $(p,n)$ are points and normals.

        Args:
            geometry (mouette.Mesh): the input geometry.
            n_points (int): number of considered (point,normal) pairs.
        """
        super().__init__()
        dim_in = geometry.dim
        
        if dim_in==2:
            points, normals = sample_points_and_normals2D(geometry, n_points)
        else:
            points,normals = M.sampling.sample_surface(geometry, n_points, return_normals=True)
        normals = torch.FloatTensor(normals)
        normals = nn.functional.normalize(normals, dim=1) # normals should be of unit norm
        self.normals = nn.Parameter(normals, requires_grad=False)
        self.points = nn.Parameter(torch.FloatTensor(points), requires_grad=False)

    def forward(self, x):
        c = torch.unsqueeze(x, 1) - self.points
        return torch.sum(nn.functional.normalize(self.normals, dim=1) * c, dim=-1) 
    



class PointDistanceEncoding(nn.Module):
    def __init__(self, geometry, n_points: int, sample_on_surface:bool = True):
        """Encodes the input point x with its l2 distance to sampled points $p$ in space:

        $$x \mapsto ||x-p||_2$$

        Args:
            geometry (mouette.Mesh): the input geometry.
            n_points (int): number of considered points.
            sample_on_surface (bool, optional): whether to sample the points on the geometry's surface, or at random in space. Defaults to True.
        """
        super().__init__()
        dim_in = geometry.dim
        
        if sample_on_surface:
            if dim_in == 2:
                points = M.sampling.sample_polyline(geometry, n_points)[:,:2]
            elif dim_in == 3:
                points = M.sampling.sample_surface(geometry, n_points)

        else:
            domain = M.geometry.AABB([-1.2]*dim_in, [1.2]*dim_in)
            points = M.sampling.sample_AABB(domain, n_points)
        self.points = nn.Parameter(torch.FloatTensor(points), requires_grad=False)

    def forward(self, x):
        c = torch.unsqueeze(x, 1) - self.points
        return torch.norm(c, dim=-1)


class GaussianEncoding(PointDistanceEncoding):
    def __init__(self, geometry, n_points: int, sample_on_surface:bool = True, stdv=1.):
        """ Variant of the PointDistanceEncoding where distance are then fed into a Gaussian kernel:

        $$x \mapsto e^{ - \\frac{||x-p||^2}{\sigma}}$$

        Args:
            geometry (mouette.Mesh): the input geometry.
            n_points (int): number of considered points.
            sample_on_surface (bool, optional): whether to sample the points on the geometry's surface, or at random in space. Defaults to True.
        """
                
        super().__init__(geometry, n_points, sample_on_surface)
        self.stdv = stdv

    def forward(self,x):
        c = torch.unsqueeze(x, 1) - self.points
        norm2 = torch.sum(c*c, dim=-1)
        return torch.exp( -norm2/self.stdv)