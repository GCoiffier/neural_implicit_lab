import pytest
from data.test_data import *
from implicitlab.data import GeometryType

@pytest.mark.parametrize("geometry_object, geometry_type", [
    (dragon(), GeometryType.POLYLINE_2D),
    (axes(), GeometryType.POINT_CLOUD_3D),
    (sphere(), GeometryType.SURFACE_MESH_3D),
    (dauphin2d_surface(), GeometryType.POLYLINE_2D),
    (dauphin2d_pl(), GeometryType.POLYLINE_2D),
    (dauphin2d_points(), GeometryType.POINT_CLOUD_2D),
    (rolling_knot(), GeometryType.POLYLINE_3D)
])
def test_geometry_type_at_import(geometry_object, geometry_type):
    assert geometry_object.geom_type == geometry_type