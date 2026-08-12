import torch
from torch.nn import functional as F
from .base import Trainer, TrainingConfig
from ..losses import HKRLoss

class hKRTrainer(Trainer):
    """Trainer minimizing the hinge-Kantorovitch-Rubinstein loss.

    Warning:
        This trainer only yields valid result when used on a Lipschitz architecture

    References:
        [1] Achieving robustness in classification using optimal transport with hinge regularization, Serrurier et al, 2021
        [2] 1-Lipschitz Neural Distance Fields, Coiffier and Béthune, 2024
    """
    def __init__(self, config : TrainingConfig, margin: float = 1e-2, lmbd: float = 100., test_mode="sdf"):
        """
        Args:
            config (TrainingConfig): training configuration hyperparameters.
            margin (float, optional): margin (m) parameter for the hKR loss. Defaults to 1e-2.
            lmbd (float, optional): lambda parameter of the hKR loss. Defaults to 100..
            test_mode (str, optional): which loss to compute for test batches. Choices are [none, sdf, hkr]. Defaults to "sdf".
        """
        super().__init__(config)
        self.lossfun = HKRLoss(margin, lmbd)
        self.testlossfun = {
            "none" : None,
            "sdf" : torch.nn.MSELoss(),
            "hkr" : HKRLoss(margin, lmbd)
        }.get(test_mode.lower(), None)

    def forward_test_batch(self, data, model):
        if self.testlossfun is None: return None
        X,Y_target = data
        Y = model(X)
        return torch.sum(self.testlossfun(Y, Y_target))

    def forward_train_batch(self, data, model):
        X,occ = data
        X.requires_grad = True
        Y = model(X)
        return torch.sum(self.lossfun(occ*Y))