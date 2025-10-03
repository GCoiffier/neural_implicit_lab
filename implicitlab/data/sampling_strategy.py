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

    def __init__(self, strategies: list, weights: list = None):
        """_summary_

        Args:
            strategies (list): _description_
            weights (list, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
        """
        super().__init__()
        assert len(strategies)>1
        self.strats: list = strategies
        if weights is None:
            n = len(self.strats)
            self.w = np.full(n, 1/n)
        else:
            self.w = np.asarray(weights)
            if np.min(self.w)<=0:
                raise Exception("weights should all be positive.")
            self.w /= np.sum(self.w) # normalize to [0;1]

    def sample(self, n_pts: int):
        sampled = []
        for i,strat in enumerate(self.strats):
            sampled.append(strat.sample(int(n_pts*self.w[i])))
        return np.concatenate(sampled)

#######################################################################################

class UniformBox(SamplingStrategy):

    def __init__(self, geom_object : M.mesh.Mesh, domain : M.geometry.AABB = None):
        """_summary_

        Args:
            geom_object (M.mesh.Mesh): _description_
            domain (M.geometry.AABB, optional): _description_. Defaults to None.
        """
        if domain is None:
            self.domain = M.geometry.AABB([-1.5]*geom_object.dim, [1.5]*geom_object.dim)
        else:
            self.domain : M.geometry.AABB = domain

    def sample(self, n_pts: int):
        return M.sampling.sample_AABB(self.domain, n_pts)

#######################################################################################

class Gaussian(SamplingStrategy):

    def __init__(self, dim: int, stdv: float):
        """_summary_

        Args:
            dim (int): _description_
            stdv (float): _description_
        """
        super().__init__()
        self.dim: int = dim
        self.stdv: float = stdv
    
    def sample(self, n_pts: int):
        return np.random.normal(0., self.stdv, size=self.dim*n_pts).reshape((n_pts, self.dim))

#######################################################################################

class NearGeometryGaussian(SamplingStrategy):
    def __init__(self, geom_object: M.mesh.Mesh, stdv:float=1e-2):
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

class Density(SamplingStrategy):
    
    def __init__(self, density_fun):
        """_summary_

        Args:
            density_fun (Callable): _description_
        """
        super().__init__()
        self.density = density_fun


    def sample(self, n_pts: int):
        return np.zeros(n_pts)