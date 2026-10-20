import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..implicit_model import ImplicitModel
from mouette.geometry import AABB


###################### Dense Grid Encoding ####################################################################

def smoothstep(x):
    return torch.pow(x, 2)*(3-2*x)


class _DenseGrid(nn.Module):
    """
    Dense feature grid of (n_cells+1)^dim vertices over an axis-aligned bounding box domain.
    Values are (multi)linearly interpolated with a single fused F.grid_sample kernel (forward and backward).
    Points outside the domain take the value of the closest boundary point.
    """

    def __init__(self, dim, n_cells, n_features, smooth=False):
        super().__init__()
        self.dim = dim
        self.n_cells = n_cells
        self.n_features = n_features
        self.smooth = smooth

        # channels-first layout expected by grid_sample: (1, F, [z,] y, x)
        grid_values = torch.zeros((1, n_features) + (n_cells+1,)*dim)
        grid_values.uniform_(-1e-3, 1e-3)
        self.grid_values = nn.Parameter(grid_values, requires_grad=True)

    def forward(self, x):
        if self.smooth:
            # remapping the local coordinates by a smoothstep turns linear into smoothstep interpolation
            t = (x + 1) * (0.5*self.n_cells)
            c = torch.floor(t.detach()).clamp_(0, self.n_cells-1)
            x = (c + smoothstep(t - c)) * (2/self.n_cells) - 1
        x = x.view((1,)*self.dim + (-1, self.dim))
        out = F.grid_sample(self.grid_values, x.to(self.grid_values.dtype), mode="bilinear", padding_mode="border", align_corners=True)
        return out.view(self.n_features, -1).t()



class MultiResDenseGridEncoding(ImplicitModel):
    def __init__(
        self,
        input_dim: int, 
        domain: AABB = None, 
        n_levels: int = 4,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 128,
        smooth: bool = False
    ):
        """
        Multiresolution encoding using dense grid of parameters.
        
        The resolutions of the grids are determined as logarithmically increasing between the base resolution and the finest resolution. For instance, the defaults parameters generate 4 grids from resolutions 16 to 128, which generates grids of resolution 16, 32, 64 and 128.

        Warning:
            The number of parameters can quickly blow up if large resolutions are used. The default finest_resolution of 128 already reaches in the millions. For better scaling, see the `MultiResHashGridEncoding` class.

        Args:
            input_dim (int): dimension of the input. This class only supports 2D or 3D inputs.
            domain (AABB, optional): domain of the grid defined as an axis-aligned bounding box. If not provided, the domain will be [0,1]^dim. Defaults to None.

            n_levels (int, optional): number of grid levels. Defaults to 4.
            n_features_per_level (int, optional): number of features per grid level. Defaults to 2.
            base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
            finest_resolution (int, optional): finest grid resolution. Defaults to 128.
            smooth (bool, optional): whether to consider a smoothstep interpolation instead of a linear one. Defaults to False.
        """
        
        super().__init__()
        self.input_dim = input_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        self.domain = domain if domain is not None else AABB.unit_cube(self.input_dim)
        
        # affine map from the domain to [-1,1]^dim (align_corners=True: -1 / 1 are the boundary vertices)
        mini = torch.tensor(np.asarray(domain.mini, dtype=np.float64)[:input_dim])
        span = torch.tensor(np.asarray(domain.span, dtype=np.float64)[:input_dim])
        self.register_buffer("scale", (2/span).float(), persistent=False)
        self.register_buffer("shift", (-1 - 2*mini/span).float(), persistent=False)

        b = math.exp((math.log(finest_resolution) - math.log(base_resolution))/(n_levels-1))

        levels = []
        for level_idx in range(n_levels):
            resolution = math.floor(base_resolution * (b ** level_idx))
            levels.append(_DenseGrid(
                dim = input_dim,
                n_cells = resolution,
                n_features=n_features_per_level,
                smooth=smooth
            ))
        self.levels = nn.ModuleList(levels)

    @property
    def output_dim(self):
        return self.n_levels * self.n_features_per_level
    
    def forward(self, x):
        x_scaled = x * self.scale + self.shift
        return torch.concatenate([_grid(x_scaled) for _grid in self.levels],dim=1)



###################### Grid Encoding with hash map #############################################################
# This code is adapted from https://github.com/Ending2015a/hash-grid-encoding/tree/master (MIT License)

@torch.no_grad()
def fast_hash(ind: torch.Tensor, primes: torch.Tensor, hashmap_size: int):
  """Hashing function from:
  https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L76-L92
  """
  d = ind.shape[-1]
  ind = (ind * primes[:d]) & 0xffffffff  # uint32
  for i in range(1, d):
    ind[..., 0] ^= ind[..., i]
  return ind[..., 0] % hashmap_size


class _HashGrid(nn.Module):
  
    PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    def __init__(
        self,
        dim: int,
        n_features: int,
        hashmap_size: int,
        resolution: int,
        dense: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.resolution = resolution
        # dense levels index the (resolution+1)^dim vertices directly (no collisions)
        self.dense = dense

        # you can add more primes for supporting more dimensions
        assert self.dim <= len(_HashGrid.PRIMES), \
            f"HashGrid only supports < {len(_HashGrid.PRIMES)}-D inputs"

        # create look-up table
        self.embedding = nn.Embedding(hashmap_size, n_features)
        nn.init.uniform_(self.embedding.weight, a=-1e-4, b=1e-4)

        primes = torch.tensor(_HashGrid.PRIMES, dtype=torch.int64)
        self.register_buffer('primes', primes, persistent=False)
        strides = torch.tensor([(resolution+1)**i for i in range(dim)], dtype=torch.int64)
        self.register_buffer('strides', strides, persistent=False)

        # create interpolation binary mask
        n_neigs = 1 << self.dim
        neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
        dims = np.arange(self.dim, dtype=np.int64).reshape((1, -1))
        bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool) # (neig, dim)
        self.register_buffer('bin_mask', bin_mask, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: (b..., dim), torch.float32, range: [0, 1]
        bdims = len(x.shape[:-1])
        x = x * self.resolution
        # floor (not truncation toward zero) and keep x == 1 inside the last cell
        xi = torch.floor(x).long().clamp(0, self.resolution-1)
        xf = x - xi.float().detach()
        xi = xi.unsqueeze(dim=-2) # (b..., 1, dim)
        xf = xf.unsqueeze(dim=-2) # (b..., 1, dim)
        # to match the input batch shape
        bin_mask = self.bin_mask.reshape((1,)*bdims + self.bin_mask.shape) # (1..., neig, dim)
        # get neighbors' indices and weights on each dim
        inds = torch.where(bin_mask, xi, xi+1) # (b..., neig, dim)
        ws = torch.where(bin_mask, 1-xf, xf) # (b...., neig, dim)
        # aggregate neighbors' interp weights
        w = ws.prod(dim=-1, keepdim=True) # (b..., neig, 1)
        # hash neighbors' id and look up table
        if self.dense:
            hash_ids = (inds * self.strides).sum(dim=-1) # (b..., neig)
        else:
            hash_ids = fast_hash(inds, self.primes, self.hashmap_size) # (b..., neig)
        neig_data = self.embedding(hash_ids) # (b..., neig, feat)
        return torch.sum(neig_data * w, dim=-2) # (b..., feat)



class MultiResHashGridEncoding(ImplicitModel):
    """Multi-resolution hash encoding from Nvidia's instant NGP project.
    Instead of considering dense grids, considers a fixed-size hashmap of feature values at each resolution, enabling higher resolution without exploding the parameter count.

    References:
        [1] _Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_, Müller et al., 2022  
        [2]  [https://nvlabs.github.io/instant-ngp](https://nvlabs.github.io/instant-ngp)  
        [3]  [https://github.com/Ending2015a/hash-grid-encoding/tree/master](https://github.com/Ending2015a/hash-grid-encoding/tree/master)  
    """
    def __init__(
        self,
        input_dim: int,
        domain: AABB = None,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        """
        Args:
            dim (int): dimension of the input (usually 2 or 3). This class supports at most 7D data.
            domain (AABB, optional): domain of the grid defined as an axis-aligned bounding box. If not provided, the domain will be [0,1]^dim. Defaults to None.
            n_levels (int, optional): number of grid levels. Defaults to 16.
            n_features_per_level (int, optional): number of features per grid level.
                Defaults to 2.
            log2_hashmap_size (int, optional): maximum size of the hashmap of each
                level in log2 scale. According to [1], this value can be set to
                14 ~ 24 depending on your problem size. Defaults to 15.
            base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
            finest_resolution (int, optional): finest grid resolution. According to [1], this value can be set to 512 ~ 524288. Defaults to 512.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        self.domain = domain if domain is not None else AABB.unit_cube(self.input_dim)
        self.register_buffer("mini", torch.tensor(domain.mini, dtype=torch.float32), persistent=False)
        self.register_buffer("span", torch.tensor(domain.span, dtype=torch.float32), persistent=False)
        
        # from paper [1] eq (3)
        b = math.exp((math.log(finest_resolution) - math.log(base_resolution))/(n_levels-1))

        levels = []
        for level_idx in range(n_levels):
            resolution = math.floor(base_resolution * (b ** level_idx))
            n_verts = (resolution+1) ** input_dim
            hashmap_size = min(n_verts, 2 ** log2_hashmap_size)
            levels.append(_HashGrid(
                dim = input_dim,
                n_features = n_features_per_level,
                hashmap_size = hashmap_size,
                dense = n_verts <= 2 ** log2_hashmap_size,
                resolution = resolution
            ))
        self.levels = nn.ModuleList(levels)

    @property
    def output_dim(self):
        return self.n_levels * self.n_features_per_level

    def forward(self, x: torch.Tensor):
        x01 = ((x - self.mini) / self.span).clamp(0., 1.) # map input to [0,1]^d
        return torch.cat([level(x01) for level in self.levels], dim=-1)
