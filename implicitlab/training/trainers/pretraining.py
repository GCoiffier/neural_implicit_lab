from .base import Trainer, TrainingConfig
from ...utils import get_model_dim
import time
from tqdm import trange
import torch
from torch.nn import functional as F

class SphereInitializeTrainer(Trainer):
    """Trainer used as an initialization step. Trains the neural network to approximate the SDF of a nD sphere
    
    References:
        [1] SAL: Sign Agnostic Learning of Shapes From Raw Data, Atzmon and Lipman, 2020
    """

    def __init__(self, n_batches: int, config: TrainingConfig, **kwargs):
        """
        Args:
            n_batches (int): _description_
            config (TrainingConfig): _description_
        
        Additionnal Args:
            radius (float, optional): radius of the sphere to consider. Defaults to 1.   
        """
        super().__init__(config)
        self.n_batches : int = n_batches
        self.radius : float = kwargs.get("radius", 1.)

    def set_training_data(self, data, shuffle = True): return
    def set_test_data(self, data): return
    def forward_test_batch(self, data, model): return 0.
    def forward_train_batch(self, data, model): return 0.

    def train(self, model):
        self.optimizer = self.get_optimizer(model)
        dim = get_model_dim(model, self.config.DEVICE)
        for cb in self.callbacks:
            cb.callOnBeginTrain(self, model)
        t0 = time.time()
        train_loss = 0.
        for _ in trange(self.n_batches):
            self.optimizer.zero_grad() # zero the parameter gradients
            points = 3*torch.rand((self.config.BATCH_SIZE, dim), device=self.config.DEVICE) - 1.5
            val = torch.squeeze(model(points))
            gt = torch.linalg.norm(points, dim=1) - self.radius
            batch_loss = F.mse_loss(val, gt)
            batch_loss.backward()
            self.optimizer.step()
            train_loss += float(batch_loss.detach())
            for cb in self.callbacks:
                cb.callOnEndForward(self, model)
        self.metrics["train_loss"] = train_loss
        self.metrics["epoch_time"] = time.time() - t0
        for cb in self.callbacks:
            cb.callOnEndTrain(self, model)