import igl
import mouette as M
from base import FieldGenerator

def Occupancy(dim: int, *args):
    if dim==2:
        return _Occupancy2D()
    elif dim==3:
        return _Occupancy3D()

class _Occupancy2D(FieldGenerator):
    pass


class _Occupancy3D(FieldGenerator):
    pass