import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from ..losses import *
from ..callbacks import Callback

import time
from dataclasses import dataclass
from abc import abstractmethod
from typing import Iterable
import warnings

@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 100
    TEST_BATCH_SIZE: int = 5000
    N_EPOCHS : int = 100
    LEARNING_RATE : float = 1e-3
    DEVICE : str = "cpu"


class Trainer:

    def __init__(self, 
        config : TrainingConfig
    ):
        if config is None:
            print("[Trainer] No configuration received. Will run with the default parameters")
            self.config = TrainingConfig() # default parameters
        else:
            self.config : TrainingConfig = config
        print("[Trainer] Configuration:", self.config)

        if self.config.DEVICE == "cpu" and torch.cuda.is_available():
            warnings.warn("Trainer is setup to run on the CPU but a compatible GPU was detected.\nTo run training on the GPU, please specify `TrainingConfig.DEVICE` as `cuda` or use the `implicitlab.utils.get_device()` function")

        self.train_data_loader : torch.utils.data.DataLoader = None
        self.test_data_loader : torch.utils.data.DataLoader = None

        self.optimizer = None
        self.callbacks = []
        self.metrics = dict()
    
    def set_training_data(self, data: TensorDataset, shuffle: bool = True):
        if not isinstance(data, TensorDataset):
            raise Exception("Please provide a torch.utils.data.TensorDataset object to this function")
        self.train_data_loader = DataLoader(data, batch_size=self.config.BATCH_SIZE, shuffle=shuffle)

    def set_test_data(self, data: TensorDataset):
        if not isinstance(data, TensorDataset):
            raise Exception("Please provide a torch.utils.data.TensorDataset object to this function")
        self.test_data_loader = DataLoader(data, batch_size=self.config.TEST_BATCH_SIZE)

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE, momentum=0.9)
    
    def add_callbacks(self, *args):
        if len(args)==1 and isinstance(args[0],Iterable): args = args[0]
        for cb in args:
            assert(isinstance(cb, Callback))
            self.callbacks.append(cb)

    def evaluate_model(self, model):
        """Evaluates the model on the test dataset.
        Computes the mean square error between neural outputs and ground truth values
        """
        if self.test_data_loader is None: return
        test_loss = 0.
        for test_batch in self.test_data_loader:
            batch_loss = self.forward_test_batch(test_batch, model)
            test_loss += batch_loss.item()
        self.metrics["test_loss"] = test_loss
        for cb in self.callbacks:
            cb.callOnEndTest(self, model)

    @abstractmethod
    def forward_test_batch(self, data, model):
        pass
    
    @abstractmethod
    def forward_train_batch(self, data, model):
        pass

    def train(self, model):
        if self.train_data_loader is None:
            raise Exception("No training data was provided. Call the `set_training_data` before training.")
        self.optimizer = self.get_optimizer(model)
        for epoch in range(self.config.N_EPOCHS):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            train_loss = 0. # accumulated loss function over all batches for monitoring purposes
            for data in tqdm(self.train_data_loader, total=len(self.train_data_loader)):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                train_batch_loss = self.forward_train_batch(data, model)
                train_batch_loss.backward()
                train_loss += float(train_batch_loss.detach())
                self.optimizer.step()
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)