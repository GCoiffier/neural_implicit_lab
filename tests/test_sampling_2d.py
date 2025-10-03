import pytest
from data.test_data import *
import os

data_2D = [
    ("dauphin2d_pl", dauphin2d_pl()),
    ("dauphin2d_pts", dauphin2d_points()),
    ("dauphin2d_surf", dauphin2d_surface()),
    ("polyline2d", polyline_2d()),
    ("dragon", dragon())
]


@pytest.mark.parametrize("name, geometry_object", data_2D)
def test_sampling_signed_distance(name, geometry_object):
    os.makedirs("tests/output", exist_ok=True)
    try:
        sampling_strategy = implicitlab.sampling_strategy.UniformBox(geometry_object)
        field_generator = implicitlab.fields.Distance(geometry_object, signed=True, square=False)
        sampler = implicitlab.PointSampler(geometry_object, sampling_strategy, field_generator)
    except implicitlab.data.fields.base.UnsupportedGeometryFormat as e:
        assert True
        return
    pts, sdf = sampler.sample(10_000)
    pts_pc = M.mesh.from_arrays(pts)
    pts_pc.vertices.register_array_as_attribute("val", sdf)
    M.mesh.save(pts_pc, f"tests/output/{name}_sdf.geogram_ascii")
    assert pts.shape == (10_000, 2)


@pytest.mark.parametrize("name, geometry_object", data_2D)
def test_sampling_occupancy(name, geometry_object):
    os.makedirs("tests/output", exist_ok=True)
    try:
        sampling_strategy = implicitlab.sampling_strategy.UniformBox(geometry_object)
        field_generator = implicitlab.fields.Occupancy(geometry_object, v_in=-1, v_out=1., v_on=0.)
        sampler = implicitlab.PointSampler(geometry_object, sampling_strategy, field_generator)
    except implicitlab.data.fields.base.UnsupportedGeometryFormat as e:
        assert True
        return
    pts, sdf = sampler.sample(10_000)
    pts_pc = M.mesh.from_arrays(pts)
    pts_pc.vertices.register_array_as_attribute("val", sdf)
    M.mesh.save(pts_pc, f"tests/output/{name}_occ.geogram_ascii")
    assert pts.shape == (10_000, 2)
