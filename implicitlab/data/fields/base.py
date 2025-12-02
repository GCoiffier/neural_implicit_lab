import mouette as M
import numpy as np
from abc import ABC, abstractmethod

from ..geometry import GeometryType

#######################################################################################

class FieldGenerator(ABC):
    
    @abstractmethod
    def compute(self, query: np.ndarray) -> np.ndarray: 
        pass

    def compute_on(self, query: np.ndarray) -> np.ndarray:
        return self.compute(query)

#######################################################################################

class EmptyFieldGenerator(FieldGenerator):
    def compute(self, *args): return None
    def compute_on(self, *args): return None

#######################################################################################

class UnsupportedGeometryFormat(Exception):
    def __init__(self, geom_type : GeometryType):
        message = f"Geometry type '{geom_type}' is not supported for this field generator"
        super().__init__(message)