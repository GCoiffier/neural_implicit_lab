import igl
import mouette as M
import numpy as np
from scipy.spatial import KDTree

from .base import FieldGenerator, UnsupportedGeometryFormat
from .utils import pseudo_surface_from_polyline
from ..geometry import GeometryType

def Distance(geom: M.mesh.Mesh, signed: bool = True, square: bool = False):
    match geom.geom_type:
        case GeometryType.POLYLINE_2D:
            return _Distance2D(geom, signed, square)
        case GeometryType.POINT_CLOUD_2D:
            return _Distance2DPointCloud(geom, signed, square)
        case GeometryType.SURFACE_MESH_3D:
            return _Distance3D(geom, signed, square)
        case GeometryType.POLYLINE_3D:
            return _Distance3DPolyline(geom, signed, square)
        case GeometryType.POINT_CLOUD_3D:
            return _Distance3DPointCloud(geom, signed, square)
        case _:
            raise UnsupportedGeometryFormat(geom.geom_type)

#######################################################################################

class _BaseDistance(FieldGenerator):
    def __init__(self, signed, square):
        super().__init__()
        self.signed : bool = signed
        self.square : bool = square

    def _apply_distance_modifier(self, d):
        match (self.square, self.signed):
            case (True, True):
                return np.sign(d)*(d**2)
            case (True, False):
                return d**2
            case (False, True):
                return d
            case (False,False):
                return np.abs(d)
    
    def compute_on(self, query: np.ndarray) -> np.ndarray:
        return np.zeros(query.shape[0])

#######################################################################################

class _Distance2DPointCloud(_BaseDistance):
    def __init__(self,
        pc: M.mesh.PointCloud,
        signed: bool,
        square: bool
    ):
        super().__init__(signed, square)
        self.tree = KDTree(np.asarray(pc.vertices)[:,:2])
    
    def compute(self, query: np.ndarray) -> np.ndarray:
        distance = self.tree.query(query)[0]
        return self._apply_distance_modifier(distance)

#######################################################################################

class _Distance2D(_BaseDistance):
    def __init__(self, 
        polyline: M.mesh.PolyLine, 
        signed: bool, 
        square: bool
    ):
        super().__init__(signed, square)
        self.V, self.F = pseudo_surface_from_polyline(polyline)
    
    def compute(self, query: np.ndarray) -> np.ndarray: 
        if query.shape[1]==2:
            query = np.pad(query, ((0,0),(0,1))) # make the points 3D
        distance = igl.signed_distance(query, self.V, self.F)[0]
        return self._apply_distance_modifier(distance)

#######################################################################################

class _Distance3D(_BaseDistance):
    
    def __init__(self, 
        mesh: M.mesh.SurfaceMesh,
        signed: bool,
        square: bool
    ):
        super().__init__(signed, square)
        self.V = np.asarray(mesh.vertices)
        self.F = np.asarray(mesh.faces, dtype=int)

    def compute(self, query : np.ndarray) -> np.ndarray:
        distance = igl.signed_distance(query, self.V, self.F)
        return self._apply_distance_modifier(distance)


#######################################################################################

class _Distance3DPolyline(_BaseDistance):

    def __init__(self, pl: M.mesh.PolyLine, square: bool):
        super().__init__(True, square)
        self.pl = pl
        V = M.sampling.sample_polyline(self.pl, 100*len(self.pl.edges))
        self.tree = KDTree(V)
    
    def compute(self, query: np.ndarray) -> np.ndarray:
        distance = self.tree.query(query)[0]
        return self._apply_distance_modifier(distance)

#######################################################################################

class _Distance3DPointCloud(_BaseDistance):
    def __init__(self, pc : M.mesh.PointCloud, square: bool):
        super().__init__(True, square)
        self.pc = pc
        self.tree = KDTree(self.pc.vertices)
    
    def compute(self, query: np.ndarray) -> np.ndarray:
        distance = self.tree.query(query)[0]
        return self._apply_distance_modifier(distance)
    
#######################################################################################

