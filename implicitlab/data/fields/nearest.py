import mouette as M
import numpy as np
import igl
from scipy.spatial import KDTree

from .base import FieldGenerator, UnsupportedGeometryFormat
from ..geometry import GeometryType

from .utils import extrude_2D_polyline

#######################################################################################

def Nearest(geom : M.mesh.Mesh):
    match geom.geom_type:
        case GeometryType.SURFACE_MESH_3D:
            return _Nearest3D_TriangleSoup(geom)
        
        case GeometryType.POINT_CLOUD_3D:
            return _Nearest_PointCloud(geom)
        case GeometryType.POINT_CLOUD_2D:
            return _Nearest_PointCloud(geom)
        
        case GeometryType.POLYLINE_2D:
            return _Nearest2D_Polyline(geom)
        
        case _:
            raise UnsupportedGeometryFormat(geom)

#######################################################################################

class _Nearest2D_Polyline(FieldGenerator):
    
    def __init__(self, pl : M.mesh.PolyLine):
        super().__init__()
        self.V, self.F = extrude_2D_polyline(pl, 10.)

    def compute(self, query: np.ndarray) -> np.ndarray:
        return igl.signed_distance(query, self.V, self.F)[2]

#######################################################################################

class _Nearest3D_TriangleSoup(FieldGenerator):
    
    def __init__(self, mesh : M.mesh.SurfaceMesh):
        super().__init__()
        self.V = np.asarray(mesh.vertices)
        self.F = np.asarray(mesh.faces)

    def compute(self, query: np.ndarray) -> np.ndarray:
        return igl.signed_distance(query, self.V, self.F)[2]

#######################################################################################

class _Nearest_PointCloud(FieldGenerator):
    
    def __init__(self, pc : M.mesh.PointCloud):
        super().__init__()
        self.V = np.asarray(pc.vertices)
        self.tree = KDTree(self.V)

    def compute(self, query: np.ndarray) -> np.ndarray: 
        nearest_inds = self.tree.query(query)[1]
        return self.V[nearest_inds,:]


    