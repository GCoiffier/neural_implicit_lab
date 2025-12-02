import mouette as M
from .sampling_strategy import *
from .fields.base import FieldGenerator, EmptyFieldGenerator

class PointSampler:

    def __init__(self, geom_object: M.mesh.Mesh, sampling_strategy: SamplingStrategy, field_generator : FieldGenerator = None):
        self.geom_object = geom_object
        self.sampler : SamplingStrategy = sampling_strategy
        self.field_generator = field_generator if field_generator is not None else EmptyFieldGenerator()
        
        self.points : np.ndarray = None
        self.field : np.ndarray = None


    def sample(self,
        n_points: int,
        on_ratio: float = 0.01,
    ):
        on_ratio = min(max(on_ratio, 0.), 1.)
        if on_ratio<1e-14:
           self.points = self.sampler.sample(n_points)
           self.field = self.field_generator.compute(self.points)
        elif on_ratio>1-1e-14:
            self.points = self._sample_geometry(n_points)
            self.field = self.field_generator.compute_on(self.points)
        else: 
            n_on = int(on_ratio*n_points)
            pts_on = self._sample_geometry(n_on)
            field_on = self.field_generator.compute_on(pts_on)
            
            n_other = n_points - n_on
            pts_other = self.sampler.sample(n_other)
            field_other = self.field_generator.compute(pts_other)
            
            self.points = np.concatenate((pts_on, pts_other))
            self.field = np.concatenate((field_on, field_other))
        
        if self.field is None:
            return self.points
        return self.points, self.field
    

    def _sample_geometry(self, n_points) -> np.ndarray:
        match type(self.geom_object):
            case M.mesh.PointCloud:
                if n_points < len(self.geom_object.vertices):
                    which = np.random.choice(len(self.geom_object.vertices), n_points, replace=False)
                    pts_on = np.array([self.geom_object.vertices[v] for v in which])
                else:
                    pts_on = np.asarray(self.geom_object.vertices)

            case M.mesh.PolyLine:
                pts_on = M.sampling.sample_polyline(self.geom_object, n_points)

            case M.mesh.SurfaceMesh:
                pts_on = M.sampling.sample_surface(self.geom_object, n_points)
        
        if self.geom_object.dim==2:
            pts_on = pts_on[:,:2]
        
        return pts_on
    
