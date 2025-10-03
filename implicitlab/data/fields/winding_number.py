import mouette as M
import numpy as np
import igl
from scipy.spatial import KDTree

from .base import FieldGenerator, UnsupportedGeometryFormat
from ..geometry import GeometryType
from .utils import pseudo_surface_from_polyline, estimate_normals, estimate_vertex_areas

#######################################################################################

def WindingNumber(geom : M.mesh.Mesh):
    match geom.geom_type:
        case GeometryType.SURFACE_MESH_3D:
            return _WindingNumber3D_TriangleSoup(geom)
        case GeometryType.POINT_CLOUD_3D:
            return _WindingNumber3D_PointCloud(geom)
        case GeometryType.POLYLINE_2D:
            return _WindingNumber2D_Polyline(geom)
        case _:
            raise UnsupportedGeometryFormat(geom)

#######################################################################################

class _WindingNumber2D_Polyline(FieldGenerator):
    
    def __init__(self, pl : M.mesh.PolyLine):
        super().__init__()
        self.V, self.F = pseudo_surface_from_polyline(pl)

    def compute(self, query: np.ndarray) -> np.ndarray:
        return igl.fast_winding_number(self.V, self.F, query)

#######################################################################################

class _WindingNumber3D_TriangleSoup(FieldGenerator):
    
    def __init__(self, mesh : M.mesh.SurfaceMesh):
        super().__init__()
        self.V = np.asarray(mesh.vertices)
        self.F = np.asarray(mesh.faces)

    def compute(self, query: np.ndarray) -> np.ndarray:
        return igl.fast_winding_number(self.V, self.F, query)

#######################################################################################

class _WindingNumber3D_PointCloud(FieldGenerator):
    
    def __init__(self, pc : M.mesh.PointCloud):
        super().__init__()
        self.pc = pc
        self.V = np.asarray(pc.vertices)
        
        if not pc.vertices.has_attribute("normals") or not pc.vertices.has_attribute("area"):
            # precompute kdtree once
            kdtree = KDTree(pc.vertices)

        if pc.vertices.has_attribute("normals"):
            self.N = pc.vertices.get_attribute("normals").as_array(len(pc.vertices))
        else:
            print("Estimate normals")
            self.N = estimate_normals(self.V, kdtree)
            pc.vertices.register_array_as_attribute("normals", self.N)

        if pc.vertices.has_attribute("area"):
            self.A = pc.vertices.get_attribute("area").as_array(len(pc.vertices))
        else:
            print("Estimate vertex local area")
            self.A = estimate_vertex_areas(self.V, self.N, kdtree)
            pc.vertices.register_array_as_attribute("area", self.A)


    def compute(self, query: np.ndarray) -> np.ndarray: 
        return igl.fast_winding_number_for_points(self.V, self.N, self.A, query)
    

    