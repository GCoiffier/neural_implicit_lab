import igl
import mouette as M
import numpy as np

from .base import FieldGenerator, UnsupportedGeometryFormat
from .utils import pseudo_surface_from_polyline
from ..geometry import GeometryType

def Occupancy(geom : M.mesh.Mesh, v_in:float, v_out:float, v_on:float):
    """_summary_

    Args:
        geom (M.mesh.Mesh): _description_
        v_in (float): _description_
        v_out (float): _description_
        v_on (float): _description_

    Raises:
        UnsupportedGeometryFormat: _description_

    Returns:
        _type_: _description_
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

    def __init__(self, geom : M.mesh.PolyLine, v_in, v_out, v_on):
        super().__init__(v_in, v_out, v_on)
        self.geom_object = geom
    
    def compute(self, query : np.ndarray) -> np.ndarray:
        wn = igl.fast_winding_number(np.asarray(self.geom_object.vertices), np.asarray(self.geom_object.faces, dtype=int), query)
        occ = np.where(wn>=0.5, self.v_in, self.v_out)
        return occ

#######################################################################################

class _Occupancy3DPointCloud(_BaseOccupancy):
    pass