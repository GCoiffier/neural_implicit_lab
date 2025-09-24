import mouette as M
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class FieldType(Enum):
    
    OCCUPANCY_BINARY = 0 # value 0 inside the shape and 1 ouside
    OCCUPANCY_TERNARY = 1 # value -1 inside, +1 outside and 0 on the boundary
    OCCUPANCY_SIGN = 2 # value -1 inside the shape and 1 outside

    DISTANCE = 10 # distance to the closest point on the surface
    SIGNED_DISTANCE = 11 # signed distance to the closest point on the surface (negative for inside points, positive outside)
    SQUARED_DISTANCE = 12 # squared distance to the closest point on the surface

    WINDING_NUMBER = 20

    @staticmethod
    def from_string(s:str):
        arg = s.lower()

        result = {
            "occupancy_binary" : FieldType.OCCUPANCY_BINARY,
            "occupancy_ternary" : FieldType.OCCUPANCY_TERNARY,
            "occupancy_sign"   : FieldType.OCCUPANCY_SIGN,
            
            "distance" : FieldType.DISTANCE,
            "udf"      : FieldType.DISTANCE,

            "squared_distance" : FieldType.SQUARED_DISTANCE,

            "signed_distance" : FieldType.SIGNED_DISTANCE,
            "sdf"             : FieldType.SIGNED_DISTANCE,

            "winding_number" : FieldType.WINDING_NUMBER,
            "wn"             : FieldType.WINDING_NUMBER
        }.get(arg, None)
        if result is None:
            raise Exception(f"string {s} was not recognized as an attribute.")
        return result


#######################################################################################

class FieldGenerator(ABC):
    
    @abstractmethod
    def compute(self, points: np.ndarray) -> np.ndarray: 
        pass
    
#######################################################################################

class UnsupportedDimensionError(Exception):
    def __init__(self, dim):
        message = f"Dimension {dim} is not supported for this field generator"
        super().__init__(message)

class UnsupportedGeometryFormat(Exception):
    def __init__(self, geom_type):
        message = f"Geometry type '{geom_type}' is not supported for this field generator"
        super().__init__(message)