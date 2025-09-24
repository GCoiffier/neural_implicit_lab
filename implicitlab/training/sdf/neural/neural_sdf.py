from .lip_model import *
from .utils import get_device, forward_in_batches
from ..sdf import SignedDistanceField
from mouette.geometry import AABB

class NeuralDistanceField(SignedDistanceField):

    def __init__(self, file_path : str, batch_size=1000, force_cpu : bool = True):
        self.device : str = get_device(force_cpu)
        self.batch_size : int = batch_size
        self.network = load_model(file_path, self.device)
        bounding_box = AABB([-0.5,-0.5,-0.5], [0.5, 0.5, 0.5])
        super().__init__(bounding_box)

    def fun(self, x):
        return forward_in_batches(self.network, x, self.device, compute_grad=False, batch_size=self.batch_size)
    
    def fungrad(self, x):
        return forward_in_batches(self.network, x, self.device, compute_grad=True, batch_size=self.batch_size)
    
    def grad(self, x):
        return self.fungrad(x)[1]