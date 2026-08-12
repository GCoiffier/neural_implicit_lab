import torch

from .base import TrainingConfig, Trainer
from ..losses import EikonalLoss


class ImplicitSurfaceTrainer(Trainer):
    """Training a neural implicit to approximate a surface

    References:
        [1] Implicit Neural Representations with Periodic Activation Functions, Sitzmann et al., 2020
        [2] Implicit Geometric Regularization for Learning Shapes, Gropp et al., 2020
    """
    def __init__(self, 
        config : TrainingConfig
    ):
        super().__init__(config)
        self.rho = 100.
        self.weights = {
            "eikonal" : 50.,
            "on" : 7000.,
            "out" : 600.,
            "normals": 100.,
        }
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts, normals = data
        pts.requires_grad = True
        Y_on = model(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        pts_out = 3*torch.rand_like(pts)-1.5
        pts_out.requires_grad = True
        Y_out = model(pts_out)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(Y_out)))

        grad = torch.autograd.grad(Y_on, pts, grad_outputs=torch.ones_like(Y_on), create_graph=True)[0]
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad, normals)
        
        batch_loss += self.weights["eikonal"] * EikonalLoss()(pts_out, Y_out)        
        return batch_loss