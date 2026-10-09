import igl
import numpy as np

from .base import FieldGenerator


class Constant(FieldGenerator):
    
    def __init__(self, value:float):
        """Constant field generator. Associates the same value at every point.

        Args:
            value (float): which value to output at every position in space
        """
        super().__init__()
        self.val = value

    def compute(self, query : np.ndarray) -> np.ndarray:
        return np.full(query.shape[0], self.val)


class CustomFunction(FieldGenerator):
    
    def __init__(self, fun, fun_on = None):
        """Custom function field generation. Considers a function f over space and associates to each point x the value f(x)

        Args:
            fun (Callable): the function to be called.
            fun_on (Callable, optional): a specific function to be called instead of `fun` if the point is on the surface. Defaults to None.
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
