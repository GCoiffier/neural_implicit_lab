import mouette as M
import numpy as np
from abc import ABC, abstractmethod

from ..geometry import GeometryType

#######################################################################################

class FieldGenerator(ABC):
    """Base class for an implicit field.
    """
    
    @abstractmethod
    def compute(self, query: np.ndarray) -> np.ndarray: 
        """_summary_

        Args:
            query (np.ndarray): the query points

        Returns:
            np.ndarray: the value of the field at each query point
        """
        pass

    def compute_on(self, query: np.ndarray) -> np.ndarray:
        """_summary_
        
        Note:
            Defining this function is optional. By default, `compute_on(query)` returns the value of `compute(query)`. Since values of a field are sometimes known for query points _on_ the geometry (for instance: a distance field has value 0 on the surface), it can be useful to avoid the field computation and overwrite this function.

        Args:
            query (np.ndarray): the query points

        Returns:
            np.ndarray: the value of the field at each query point.
        """
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