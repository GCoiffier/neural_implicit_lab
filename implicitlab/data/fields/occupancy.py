import igl
import mouette as M
import numpy as np

from .base import FieldGenerator, UnsupportedGeometryFormat
from .utils import pseudo_surface_from_polyline
from ..geometry import GeometryType

def Occupancy(geom : M.mesh.Mesh, v_in:float, v_out:float, v_on:float):
    """Computes the occupancy of the given geometry, i.e. a constant value inside and another constant value outside of the object.

    Args:
        geom (M.mesh.Mesh): geometrical object to consider.
        v_in (float): value for points inside the object.
        v_out (float): value for points outside the object.
        v_on (float): value for points on the boundary of the object.

    Raises:
        UnsupportedGeometryFormat: only makes sense for (closed) 2D polylines and 3D surface meshes.
    """
    match geom.geom_type:
        # case GeometryType.POINT_CLOUD_2D:
        #     return
        case GeometryType.POLYLINE_2D:
            return _Occupancy2D(geom, v_in, v_out, v_on)
        # case GeometryType.POINT_CLOUD_3D:
        #     return _Occupancy3DPointCloud(geom, v_in, v_out, v_on)
        case GeometryType.SURFACE_MESH_3D:
            return _Occupancy3D(geom, v_in, v_out, v_on)
        case _:
            raise UnsupportedGeometryFormat(geom.geom_type)

#######################################################################################

class _BaseOccupancy(FieldGenerator):
    def __init__(self, v_in, v_out, v_on):
        super().__init__()
        self.v_in  : float = v_in
        self.v_out : float = v_out
        self.v_on  : float = v_on

    def compute_on(self, query: np.ndarray) -> np.ndarray:
        return np.full(query.shape[0], self.v_on)

#######################################################################################

class _Occupancy2D(_BaseOccupancy):
    def __init__(self, geom : M.mesh.PolyLine, v_in, v_out, v_on):
        super().__init__(v_in, v_out, v_on)
        self.geom_object = geom
        self.V, self.F = pseudo_surface_from_polyline(geom)

    def compute(self, query : np.ndarray) -> np.ndarray:
        wn = igl.fast_winding_number(self.V, self.F, query)
        occ = np.where(wn>=0.5, self.v_in, self.v_out)
        return occ

#######################################################################################

class _Occupancy3D(_BaseOccupancy):

    def __init__(self, geom : M.mesh.PolyLine, v_in, v_out, v_on, threshold: float =0.5):
        super().__init__(v_in, v_out, v_on)
        self.geom_object = geom
        self.threshold = threshold
    
    def compute(self, query : np.ndarray) -> np.ndarray:
        wn = igl.fast_winding_number(np.asarray(self.geom_object.vertices), np.asarray(self.geom_object.faces, dtype=int), query)
        mean_wn = np.mean(wn)
        if mean_wn>0:
            occ = np.where(wn>=self.threshold, self.v_in, self.v_out)
        else:
            occ = np.where(wn<=-self.threshold, self.v_in, self.v_out)
        return occ