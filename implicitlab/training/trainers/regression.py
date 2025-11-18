import torch
from torch.nn import functional as F
from .base import Trainer
from ..losses import EikonalLoss

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
        batch_loss_eik = self.eikonal_weight * EikonalLoss(x_rdm, y_rdm)
        
        return batch_loss_fit + batch_loss_eik

        