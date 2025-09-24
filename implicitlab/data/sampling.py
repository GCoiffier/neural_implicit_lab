import mouette as M
from strategy import *
from implicitlab.data.fields.base import *
from utils import get_data_dimensionnality
import fields

class PointSampler:

    def __init__(self, geom_object: M.mesh.Mesh, sampling_strategy: SamplingStrategy, field_type : FieldType):
        self.dim = get_data_dimensionnality(geom_object)
        if self.dim==2 and isinstance(geom_object, M.mesh.SurfaceMesh):
            # input mesh is a solid 2D mesh -> we are only interested in its boundary polyline
            self.geom_object = M.processing.extract_boundary_of_surface(geom_object)[0]
        else:
            self.geom_object = geom_object

        self.sampler : SamplingStrategy = sampling_strategy
        if isinstance(field_type, fields.FieldGenerator):
            self.field_generator = field_type
        elif isinstance(field_type, FieldType):
            self.field_generator = self._get_field_generator(field_type)
        
        self.points : np.ndarray = None
        self.field : np.ndarray = None

    def _get_field_generator(self, ftype : FieldType):
        match ftype:
            case FieldType.OCCUPANCY_BINARY:
                self.field_generator = fields.Occupancy(self.dim)
            case FieldType.OCCUPANCY_SIGN:
                self.field_generator = fields.Occupancy(self.dim)
            case FieldType.OCCUPANCY_TERNARY:
                self.field_generator = fields.Occupancy(self.dim)

            case FieldType.WINDING_NUMBER:
                self.field_generator = fields.WindingNumber(self.dim)
            
            case FieldType.SIGNED_DISTANCE:
                self.field_generator = fields.Distance(self.dim, signed=True, square=False)
            case FieldType.DISTANCE:
                self.field_generator = fields.Distance(self.dim, signed=False, square=False)
            case FieldType.SQUARED_DISTANCE:
                self.field_generator = fields.Distance(self.dim, signed=True, square=True)


    def sample(self,
        n_points: int,
        on_ratio: float = 0.01,
    ):
        n_on = int(on_ratio*n_points)
        pts_on = self._sample_geometry(n_on)
        field_on = np.zeros(n_on)
        
        n_other = n_points - n_on
        pts_other = self.sampler.sample(n_other)
        field_other = self.field_generator.compute(pts_other)

        self.points = np.concatenate((pts_on, pts_other))
        self.field = np.concatenate((field_on, field_other))
        return self.points, self.field
    

    def _sample_geometry(self, n_points) -> np.ndarray:
        match type(self.geom_object):
            case M.mesh.PointCloud:
                if n_points > len(self.geom_object.vertices):
                    which = np.random.choice(len(self.geom_object.vertices), n_points, replace=False)
                    pts_on = np.array((self.geom_object.vertices[v] for v in which))
                else:
                    pts_on = np.asarray(self.geom_object.vertices)

            case M.mesh.PolyLine:
                pts_on = M.sampling.sample_polyline(self.geom_object, n_points)

            case M.mesh.SurfaceMesh:
                pts_on = M.sampling.sample_surface(self.geom_object, n_points)
        return pts_on
    
