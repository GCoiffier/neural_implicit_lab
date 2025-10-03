import torch
from torch.nn import functional as F
from .base import Trainer

class SimpleRegressionTrainer(Trainer):
    def __init__(self, config, lossfun):
        super().__init__(config)
        self.lossfun = lossfun

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return self.lossfun(Y, Y_target)

    def forward_train_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return self.lossfun(Y, Y_target)
    

class RegressionEikonalTrainer(Trainer):
    def __init__(self, config, eikonal_weight:float = 0.01):
        super().__init__(config)
        self.eikonal_weight: float = eikonal_weight

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return F.mse_loss(Y, Y_target)

    def forward_train_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        batch_loss_fit = F.mse_loss(Y, Y_target)
        x_rdm = 2*torch.rand_like(X)-1 # between -1 and 1
        x_rdm.requires_grad = True
        y_rdm = model(x_rdm)
        batch_grad = torch.autograd.grad(y_rdm, x_rdm, grad_outputs=torch.ones_like(y_rdm), create_graph=True)[0]
        batch_grad_norm = batch_grad.norm(dim=-1)
        batch_loss_eik = self.eikonal_weight * F.mse_loss(batch_grad_norm, torch.ones_like(batch_grad_norm))
        return batch_loss_fit + batch_loss_eik

        