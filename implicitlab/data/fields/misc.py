import igl
import numpy as np

from .base import FieldGenerator


class Constant(FieldGenerator):
    
    def __init__(self, value:float):
        """_summary_

        Args:
            value (float): _description_
        """
        super().__init__()
        self.val = value

    def compute(self, query : np.ndarray) -> np.ndarray:
        return np.full(query.shape[0], self.value)


class CustomFunction(FieldGenerator):
    
    def __init__(self, fun, fun_on = None):
        """_summary_

        Args:
            fun (Callable): _description_
            fun_on (Callable, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.fun = np.vectorize(fun, signature="(n)->()")
        self.fun_on = None
        if fun_on is not None:
            self.fun_on = np.vectorize(fun_on,  signature="(n)->()")

    def compute(self, query: np.ndarray) -> np.ndarray:
        return self.fun(query)

    def compute_on(self, query: np.ndarray) -> np.ndarray:
        if self.fun_on is None:
            return self.fun(query)
        return self.fun_on(query)
    
    def _get_fun_dimensionnality(self) -> int:
        pt2D = np.zeros((1,2))
        pt3D = np.zeros((1,3))
        try:
            _ = self.fun(pt2D)
            return 2
        except: pass
        try:
            _ = self.fun(pt3D)
            return 3
        except: pass
