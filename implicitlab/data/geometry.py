import mouette as M
import meshio
import numpy as np
from enum import Enum

class GeometryType(Enum):

    POINT_CLOUD_2D = 0
    POINT_CLOUD_3D = 1

    POLYLINE_2D = 2
    POLYLINE_3D = 3

    SURFACE_MESH_2D = 4
    SURFACE_MESH_3D = 5

    @classmethod
    def of_geometry_data(cls, geom : M.mesh.Mesh):
        dim = get_data_dimensionnality(geom)
        match (dim, type(geom)):
            case (2, M.mesh.PointCloud):
                return cls.POINT_CLOUD_2D
            case (3, M.mesh.PointCloud):
                return cls.POINT_CLOUD_3D
            case (2, M.mesh.PolyLine):
                return cls.POLYLINE_2D
            case (3, M.mesh.PolyLine):
                return cls.POLYLINE_3D
            case (2, M.mesh.SurfaceMesh):
                return cls.SURFACE_MESH_2D
            case (3, M.mesh.SurfaceMesh):
                return cls.SURFACE_MESH_3D
            case _:
                raise Exception("Geometry type not recognized.")

    @property
    def dim(self) -> int:
        if self in (GeometryType.POINT_CLOUD_2D, GeometryType.POLYLINE_2D, GeometryType.SURFACE_MESH_2D):
            return 2
        elif self in (GeometryType.POINT_CLOUD_3D, GeometryType.POLYLINE_3D, GeometryType.SURFACE_MESH_3D):
            return 3

def is_normalized(mesh : M.mesh.Mesh) -> bool:
    coords = np.asarray(mesh.vertices)
    return np.min(coords) >= -1 and np.max(coords) <= 1. 

def get_data_dimensionnality(mesh : M.mesh.Mesh) -> int:
    bb = M.geometry.AABB.of_mesh(mesh)
    span = bb.span
    if span.size==2: return 2
    return np.sum((span/np.max(span))>1e-8)

def prepare_geometry(geom : M.mesh.Mesh):
    geom_type = GeometryType.of_geometry_data(geom)
    geom = M.transform.normalize(geom)
    geom.geom_type = geom_type
    geom.dim = geom_type.dim

    match geom_type:
        case GeometryType.SURFACE_MESH_3D:
            geom = M.mesh.triangulate(geom)
            geom.geom_type = GeometryType.SURFACE_MESH_3D
            geom.dim = 3
            
        case GeometryType.SURFACE_MESH_2D:
            geom = M.processing.extract_boundary_of_surface(geom)[0]
            geom.geom_type = GeometryType.POLYLINE_2D
            geom.dim = 2
        case _:
            # Nothing to do in other cases
            pass
    return geom

def load_geometry(file_path : str):
    try:
        raise NotImplementedError
        geom = meshio.read(file_path)
    except:
        geom = M.mesh.load(file_path)
    return prepare_geometry(geom)