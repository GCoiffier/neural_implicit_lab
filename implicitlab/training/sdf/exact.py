from .sdf import SignedDistanceField
import numpy as np
from mouette.geometry import AABB

class SphereSDF(SignedDistanceField):

    def __init__(self, center : np.ndarray, radius : float):
        self.center = center
        self.radius = radius
        r = np.full_like(self.center, radius)
        bb = AABB(center - r, center + r)
        super().__init__(bb)

    def fun(self, x):
        return np.linalg.norm(x-self.center) - self.radius
    
    def grad(self, x):
        return x-self.center