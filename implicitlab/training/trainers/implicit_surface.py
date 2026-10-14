import torch

from .base_trainer import TrainingConfig, Trainer
from ..losses import EikonalLoss


class ImplicitSurfaceTrainer(Trainer):
    """Training a neural implicit to approximate a shape as its zero level set. The loss functions and hyperparameters are adapted from [1] and [2]. Given a neural network $f$, The method optimizes for four loss functions:

    $$\\mathcal{L}_{\\text{on}}[f]  = \\int_M |f(x)| dx$$

    $$\\mathcal{L}_{\\text{normal}}[f] = \\int_M  ||\\nabla f(x) - n(x)||\\;dx$$

    $$\\mathcal{L}_{\\text{eikonal}}[f] = \\int_D (|| \\nabla f(x)|| - 1)^2\\;dx$$

    $$\\mathbb{L}_{\\text{out}}[f] = \\int_D \\exp(-\\rho\\,|f(x)|)\\;dx$$

    This trainer assumes that the dataset is made of (points, normals) pairs which are sampled _on_ the shape, for the evaluation of $$\\mathcal{L}_{\\text{on}}$ and $\\mathcal{L}_{\\text{normal}}$. The two other loss functions are evaluated for every batch at random positions in a [-1.5, 1.5]^d box.

    Note:
        The `forward_test_batch` method is not implemented and does nothing.

    References:
        [1] _Implicit Neural Representations with Periodic Activation Functions_, Sitzmann et al., 2020  
        [2] _Implicit Geometric Regularization for Learning Shapes_, Gropp et al., 2020  

    Attributes:
        rho (float): value of rho in the outside loss. Defaults to 100.
        weights (dict): dictionnary of weights that are associated with each loss function. Keys and default values are:
            - "on" : 100.
            - "out" : 5.
            - "normal": 1.
            - "eikonal" : 1.

    """
    def __init__(self, 
        config : TrainingConfig
    ):
        """
        Args:
            config (TrainingConfig): The training hyperparameters.
        """
        super().__init__(config)
        self.rho = 100.
        self.weights = {
            "eikonal" : 1.,
            "on" : 100.,
            "out" : 5.,
            "normals": 1.,
        }
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts_on, normals = data
        n_pts_on = pts_on.shape[0]
        pts_out = 3*torch.rand_like(pts_on)-1.5
        pts_all = torch.cat((pts_on, pts_out))
        pts_all.requires_grad = True
        value = model(pts_all)
        grad = torch.autograd.grad(value, pts_all, grad_outputs=torch.ones_like(value), create_graph=True)[0]

        value_on, value_out = value[:n_pts_on], value[n_pts_on:]
        grad_on, grad_out = grad[:n_pts_on], grad[n_pts_on:]

        batch_loss = self.weights["on"] * torch.mean(torch.abs(value_on))
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad_on, normals)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(value_out)))
        batch_loss += self.weights["eikonal"] * EikonalLoss(pts_out, value_out, grad_out)        
        return batch_loss