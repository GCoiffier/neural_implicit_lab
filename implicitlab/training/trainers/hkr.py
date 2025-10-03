import torch
from torch.nn import functional as F
from .base import Trainer
from ..loss import LossHKR

class hKRTrainer(Trainer):

    def __init__(self, config, margin, lmbd=0.1, test_mode="sdf"):
        super().__init__(config)
        self.lossfun = LossHKR(margin, lmbd)
        if test_mode.lower()=="sdf":
            self.testlossfun = torch.nn.MSELoss()
        elif test_mode.lower()=="hkr":
            self.testlossfun = LossHKR(margin, lmbd)

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        X,occ = data
        Y = model(X)
        return torch.sum(self.lossfun(occ*Y))
    