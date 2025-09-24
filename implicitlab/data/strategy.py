from abc import ABC, abstractmethod
import numpy as np

class SamplingStrategy(ABC):

    @abstractmethod
    def sample(self, n_pts: int) -> np.ndarray:
        return
    

class UniformBox(SamplingStrategy):
    
    def sample(self, n_pts: int):
        return np.zeros(n_pts)


class Gaussian(SamplingStrategy):
    
    def sample(self, n_pts: int):
        return np.zeros(n_pts)


class Density(SamplingStrategy):
    
    def sample(self, n_pts: int):
        return np.zeros(n_pts)


class Importance(SamplingStrategy):

    def sample(self, n_pts: int):
        return np.zeros(n_pts)