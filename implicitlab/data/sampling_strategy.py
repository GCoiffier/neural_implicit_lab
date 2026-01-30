from abc import ABC, abstractmethod
import numpy as np
import mouette as M

#######################################################################################

class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, n_pts: int) -> np.ndarray:
        return
    
#######################################################################################

class CombinedStrategy(SamplingStrategy):

    def __init__(self, strategies: list, weights: list = None, shuffle: bool = True):
        """Combines different sampling strategies into one.

        Args:
            strategies (list): list of SamplingStrategy objects to call for sampling.
            weights (list, optional): relative weights of the sampling strategies. Weights [1,2] means that twice as many points will be sampled from the second strategy. If not provided, sampling is uniform. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the output points

        Raises:
            Exception: Fails if weights are not all positive.
        """
        super().__init__()
        assert len(strategies)>1
        self.strats: list = strategies
        self.shuffle = shuffle
        if weights is None:
            n = len(self.strats)
            self.w = np.full(n, 1/n)
        else:
            self.w = np.asarray(weights, dtype=float)
            if np.min(self.w)<=0:
                raise Exception("weights should all be positive.")
            self.w /= np.sum(self.w) # normalize to [0;1]

    def sample(self, n_pts: int):
        sampled = []
        for i,strat in enumerate(self.strats):
            sampled.append(strat.sample(int(n_pts*self.w[i])))
        sampled = np.concatenate(sampled)
        if self.shuffle: np.random.shuffle(sampled)
        return sampled

#######################################################################################

class UniformBox(SamplingStrategy):

    def __init__(self, geom_object : M.mesh.Mesh, domain : M.geometry.AABB = None):
        """Uniform sampling in an axis-aligned bounding box.

        Args:
            geom_object (M.mesh.Mesh): geometry to sample. Only the `dim` attribute is accessed.
            domain (M.geometry.AABB, optional): bounding box to consider for sampling. If not provided, will take [-1.5, 1.5]^n as default. Defaults to None.
        """
        if domain is None:
            self.domain = M.geometry.AABB([-1.5]*geom_object.dim, [1.5]*geom_object.dim)
        else:
            self.domain : M.geometry.AABB = domain

    def sample(self, n_pts: int):
        return M.sampling.sample_AABB(self.domain, n_pts)

#######################################################################################

class Gaussian(SamplingStrategy):

    def __init__(self, geom_object : M.mesh.Mesh, stdv: float):
        """Multivariate Gaussian sampling with zero mean.

         Args:
            geom_object (M.mesh.Mesh): geometry to sample. Only the `dim` attribute is accessed.
            stdv (float): Standard variation of the Gaussian distribution.
        """
        super().__init__()
        self.dim: int = geom_object.dim
        self.stdv: float = stdv
    
    def sample(self, n_pts: int):
        return np.random.normal(0., self.stdv, size=self.dim*n_pts).reshape((n_pts, self.dim))

#######################################################################################

class NearGeometryGaussian(SamplingStrategy):
    def __init__(self, geom_object: M.mesh.Mesh, stdv:float=1e-2):
        """Sample points around a given object by first sampling points on the object and add a Gaussian noise.

        Args:
            geom_object (M.mesh.Mesh): geometry to sample.
            stdv (float, optional): Standard variation of the Gaussian distribution. Defaults to 1e-2.
        
        Raises:
            Exception: Fails if the function is unable to sample the given geometrical object.

        """
        super().__init__()
        self.geom_object = geom_object
        self.stdv = stdv

    def sample(self, n_pts: int):
        match type(self.geom_object):
            case M.mesh.PointCloud:
                ch = np.random.choice(self.geom_object.id_vertices, n_pts, replace=False)
                x = np.array([self.geom_object.vertices[v] for v in ch])
            case M.mesh.PolyLine:
                x = M.sampling.sample_polyline(self.geom_object, n_pts)
            case M.mesh.SurfaceMesh:
                x = M.sampling.sample_surface(self.geom_object, n_pts)
            case _:
                raise Exception("Geometry type not recognized. Something bad happened.")
        if self.geom_object.dim==2:
            x = x[:,:2]
        x += np.random.normal(0.,self.stdv, size=x.shape)
        return x

#######################################################################################

# TODO: sample according to a probability distribution function using Metropolis-Hastings algorithms
# class Density(SamplingStrategy):
    
#     def __init__(self, density_fun):
#         """_summary_

#         Args:
#             density_fun (Callable): _description_
#         """
#         super().__init__()
#         self.density = density_fun


#     def sample(self, n_pts: int):
#         return np.zeros(n_pts)