import mouette as M
from .sampling_strategies import *
from .fields.base import FieldGenerator

class PointSampler:

    def __init__(self, geom_object: M.mesh.Mesh, sampling_strategy: SamplingStrategy, *field_generators : list[FieldGenerator]):
        """The PointSampler class is responsible for generating a dataset of points to be fed to a neural implicit representation during training.

        Args:
            geom_object (M.mesh.Mesh): the input geometrical object around which points are sampled.
            sampling_strategy (SamplingStrategy): the strategy to be used for sampling points (see Sampling Strategies)
            *field_generators (FieldGenerator, optional): the data to be computed at each sampled points (see Implicit Fields). Several fields can be provided for several signals to be computed. Defaults to None (no field will be generated).
        """
        self.geom_object = geom_object
        self.sampler : SamplingStrategy = sampling_strategy
        self.field_generators = list(field_generators)
        self.points : np.ndarray = None
        self.fields : tuple = None


    def sample(self,
        n_points: int,
        on_ratio: float = 0.01,
    ):
        """Samples a given number of points according to the provided sampling strategy.

        Args:
            n_points (int): number of points to sample.
            on_ratio (float, optional): ratio of sampled points that are taken on the geometry rather than around it. Defaults to 0.01.

        Returns:
            np.ndarray, Tuple(np.ndarray): An array of sampled points and an array of the field value at these points.
        """
        on_ratio = min(max(on_ratio, 0.), 1.)
        if on_ratio<1e-14:
           self.points = self.sampler.sample(n_points)
           self.fields = tuple(field.compute(self.points) for field in self.field_generators)

        elif on_ratio>1-1e-14:
            self.points = self.sample_geometry(n_points)
            self.fields = tuple(field.compute_on(self.points) for field in self.field_generators)
        else: 
            n_on = int(on_ratio*n_points)
            pts_on = self.sample_geometry(n_on)
            fields_on = tuple(field.compute_on(pts_on) for field in self.field_generators)
            
            n_other = n_points - n_on
            pts_other = self.sampler.sample(n_other)
            fields_other = tuple(field.compute(pts_other) for field in self.field_generators)
            
            self.points = np.concatenate((pts_on, pts_other))
            if fields_on is not None and fields_other is not None:
                self.fields = tuple(np.concatenate((_on, _other)) for (_on, _other) in zip(fields_on, fields_other))
        
        if self.fields is None:
            return self.points
        elif len(self.fields)==1:
            return self.points, self.fields[0]
        return self.points, self.fields
    

    def sample_geometry(self, n_points: int) -> np.ndarray:
        """Samples a given number of points _on_ the geometrical object provided.

        Args:
            n_points (int): number of points to sample

        Returns:
            np.ndarray: array of sampled points
        """
        match type(self.geom_object):
            case M.mesh.PointCloud:
                if n_points < len(self.geom_object.vertices):
                    which = np.random.choice(len(self.geom_object.vertices), n_points, replace=False)
                    self.points = np.array([self.geom_object.vertices[v] for v in which])
                else:
                    self.points = np.asarray(self.geom_object.vertices)

            case M.mesh.PolyLine:
                self.points = M.sampling.sample_polyline(self.geom_object, n_points)

            case M.mesh.SurfaceMesh:
                self.points = M.sampling.sample_surface(self.geom_object, n_points)
        
        if self.geom_object.dim==2:
            self.points = self.points[:,:2]
        
        self.fields = tuple(field.compute_on(self.points) for field in self.field_generators)
        if not self.fields:
            return self.points       
        return self.points, self.fields