import torch
from torch.nn import functional as F
from .base import Trainer

class NeuralPullTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)
        
    def forward_test_batch(self, data, model):
        # same as train
        return self.forward_train_batch(data, model)

    def forward_train_batch(self, data, model):
        X,NN = data
        X.requires_grad = True
        Y = model(X)
        grad = torch.autograd.grad(Y, X, grad_outputs=torch.ones_like(Y), create_graph=True)[0]
        grad_normed = F.normalize(grad, dim=1)
        NN_pred = X - Y*grad_normed
        return F.mse_loss(NN_pred, NN)
    