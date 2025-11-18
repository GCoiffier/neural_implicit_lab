import IL
from IL import load_geometry
import mouette as M

##### 2D #####

def dragon():
    return load_geometry("tests/data/dragon.obj")

def dauphin2d_surface():
    return load_geometry("tests/data/dauphin_surf.obj")

def dauphin2d_pl():
    return load_geometry("tests/data/dauphin_pl.mesh")

def dauphin2d_points():
    return load_geometry("tests/data/dauphin_pts.xyz")

def polyline_2d():
    return load_geometry("tests/data/polyline_intersect.mesh")


##### 3D #####

def axes():
    return load_geometry("tests/data/axes.xyz")

def sphere():
    sph = M.procedural.sphere_fibonacci(550)
    return IL.data.prepare_geometry(sph)

def spot():
    return load_geometry("tests/data/spot.obj")

def rolling_knot():
    return load_geometry("tests/data/rolling_knot.obj")

