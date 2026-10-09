import torch

from .base_trainer import TrainingConfig, Trainer
from ..losses import EikonalLoss


class ImplicitSurfaceTrainer(Trainer):
    """Training a neural implicit to approximate a shape as its zero level set. The loss functions and hyperparameters are adapted from [1] and [2]. Given a neural network $f$, The method optimizes for four loss functions:

    $$\\mathcal{L}_{\\text{on}}[f]  = \\int_M |f(x)| dx$$

    $$\\mathcal{L}_{\\text{normal}}[f] = \\int_M  ||\\nabla f(x) - n(x)||\\;dx$$

    $$\\mathcal{L}_{\\text{eikonal}}[f] = \\int_D (|| \\nabla f(x)|| - 1)^2\\;dx$$

    $$\\mathbb{L}_{\\text{out}}[f] = \\int_D \\exp(-\\rho\,|f(x)|)\\;dx$$

    This trainer assumes that the dataset is made of (points, normals) pairs which are sampled _on_ the shape, for the evaluation of $$\\mathcal{L}_{\\text{on}}$ and $\\mathcal{L}_{\\text{normal}}$. The two other loss functions are evaluated for every batch at random positions in a [-1.5, 1.5]^d box.

    Note:
        The `forward_test_batch` method is not implemented and does nothing.

    References:
        [1] _Implicit Neural Representations with Periodic Activation Functions_, Sitzmann et al., 2020  
        [2] _Implicit Geometric Regularization for Learning Shapes_, Gropp et al., 2020  

    Attributes:
        rho (float): value of rho in the outside loss. Defaults to 100.
        weights (dict): dictionnary of weights that are associated with each loss function. Keys and default values are:
            - "on" : 700
            - "out" : 60
            - "normal": 10
            - "eikonal" : 5

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
            "eikonal" : 5.,
            "on" : 700.,
            "out" : 60.,
            "normals": 10.,
        }
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts, normals = data
        pts.requires_grad = True
        Y_on = model(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        grad_on = torch.autograd.grad(Y_on, pts, grad_outputs=torch.ones_like(Y_on), create_graph=True)[0]
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad_on, normals)

        pts_out = 3*torch.rand_like(pts)-1.5
        pts_out.requires_grad = True
        Y_out = model(pts_out)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(Y_out)))
        batch_loss += self.weights["eikonal"] * EikonalLoss(pts_out, Y_out)        
        return batch_loss