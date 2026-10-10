import mouette as M
import meshio
import numpy as np
from enum import Enum

class GeometryType(Enum):
    """Enumeration of the kinds of geometrical objects handled by the library.

    A GeometryType attribute `.type` is automatically assigned to every geometrical object loaded by the library. It specifies the nature of the object (point cloud, polyline or surface mesh) and the dimension of its ambient space (2D or 3D). Algorithms access this type to determine if it makes sense to apply a given processing to a given type.

    Attributes:
        POINT_CLOUD_2D: a set of points in the plane.
        POINT_CLOUD_3D: a set of points in space.
        POLYLINE_2D: a set of points connected by edges in the plane (typically a closed curve).
        POLYLINE_3D: a set of points connected by edges in space.
        SURFACE_MESH_2D: a planar surface mesh.
        SURFACE_MESH_3D: a surface mesh in space.
    """

    POINT_CLOUD_2D = 0
    POINT_CLOUD_3D = 1

    POLYLINE_2D = 2
    POLYLINE_3D = 3

    SURFACE_MESH_2D = 4
    SURFACE_MESH_3D = 5

    @classmethod
    def of_geometry_data(cls, geom : M.mesh.Mesh):
        """Infers the geometry type of a mesh object from its class and the dimensionality of its vertex coordinates.

        Note:
            A 3D object whose vertices all lie in a plane (for instance with a constant z coordinate) will be converted to a 2D object.

        Args:
            geom (mouette.mesh.Mesh): the geometrical object. Should be a PointCloud, a PolyLine or a SurfaceMesh.

        Raises:
            Exception: if the type of the object or its dimensionality is not supported.

        Returns:
            GeometryType: the type of the geometrical object.
        """
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
        """Dimension of the ambient space of the geometry type.

        Returns:
            int: 2 for planar geometries, 3 for geometries embedded in space.
        """
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
        case _:
            # Nothing to do in other cases
            pass
    return geom


def extract_boundary_polyline_from_2D_mesh(geometry):
    """Converts a `SURFACE_MESH_2D` into a `POLYLINE_2D` geometrical object by extracting its boundary polyline.

    If the geometry type is not `SURFACE_MESH_2D`, this function does nothing.

    Args:
        geometry (mouette.mesh.Mesh): the input geometry.

    Returns:
        geometry (mouette.mesh.Mesh): The extracted boundary polyline if the input geometry type was SURFACE_MESH_2D. Otherwise, returns the unmodified input geometry.
    """
    if geometry.type != GeometryType.SURFACE_MESH_2D:
        return geometry
    geom = M.processing.extract_boundary_of_surface(geom)[0]
    geom.geom_type = GeometryType.POLYLINE_2D
    geom.dim = 2


def load_geometry(file_path : str):
    """Loads a geometrical object from a file on the disk. The created object is a `mouette` mesh type. See [the documentation of mouette](https://gcoiffier.github.io/mouette/datastructures/SurfaceMeshes/) for further information.

    Note:
        By convention:  
        - The geometry object is normalized so that it fits the unit box [-1, 1]^d (where d is its dimension).  
        - All surface meshes are triangulated at import

    Args:
        file_path (str): path to the file. Supported formats are .obj, .mesh, .stl, .xyz and .geogram_ascii

    Returns:
        mouette.mesh.Mesh: a geometry object (whose type depend on what's been read from the file). Can be a PointCloud, a Polyline or a SurfaceMesh.
    """
    try:
        raise NotImplementedError
        geom = meshio.read(file_path)
    except:
        geom = M.mesh.load(file_path)
    return prepare_geometry(geom)