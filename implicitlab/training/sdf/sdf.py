from abc import ABC, abstractmethod
from mouette.geometry import AABB

class SignedDistanceField(ABC):

    def __init__(self, bounding_box: AABB):
        self.bb = bounding_box

    def __call__(self, *args, **kwds):
        return self.fun(*args, **kwds)

    @abstractmethod
    def fun(self, x, *args, **kwds):
        pass

    @abstractmethod
    def grad(self, x, *args, **kwds):
        pass

    def fungrad(self, x, *args, **kwds):
        return self.fun(x, *args, **kwds), self.grad(x, *args, **kwds)