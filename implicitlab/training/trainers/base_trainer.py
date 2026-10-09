import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from ..losses import *
from ..callbacks import Callback
from ..optimizers import SingleDeviceMuonWithAuxAdam, AdaBound, AdaBoundW

import time
from dataclasses import dataclass
from abc import abstractmethod
from typing import Iterable
import warnings

@dataclass
class TrainingConfig:
    """Configuration object storing hyperparameters for a `Trainer` class.

    Attributes:
        BATCH_SIZE (int): Number of samples per batch in the training data loader. Defaults to 100.
        TEST_BATCH_SIZE (int): Number of samples per batch in the test data loader. Defaults to 5000.
        N_EPOCHS (int): Number of passes over the training dataset. Defaults to 100.
        LEARNING_RATE (float): Learning rate given to the optimizer. Defaults to 1e-3.
        DEVICE (str): Torch device on which training runs ("cpu", "cuda", ...). A warning is emitted if
            set to "cpu" while a GPU is available. Defaults to "cpu".
        OPTIMIZER (str): Name of the optimizer (case-insensitive). One of "adam", "sgd", "muon", "adabound", or "custom". If "custom" is selected, the user must set `trainer.optimizer` before calling the `train` method). Unknown names fall back to Adam. Defaults to "Adam".
        NUM_DATALOADER_WORKERS (int): Number of worker processes for the train data loaders. -1 means `torch.get_num_threads()`. Defaults to -1.
    """
    BATCH_SIZE: int = 100
    TEST_BATCH_SIZE: int = 5000
    N_EPOCHS : int = 100
    LEARNING_RATE : float = 1e-3
    DEVICE : str = "cpu"
    OPTIMIZER : str = "Adam"
    NUM_DATALOADER_WORKERS: int = -1


class Trainer:
    """Base trainer class. Implements the optimization loop for a given neural implicit model.

    This class is not meant to be used directly, but to be inherited from. Subclasses need to implement the abstract methods `forward_train_batch` and `forward_test_batch`.

    Attributes:
        config (TrainingConfig): The training hyperparameters.
        train_data_loader (DataLoader): Loader over the training data. Set by `set_training_data`.
        test_data_loader (DataLoader): Loader over the test data. Set by `set_test_data`.
        optimizer (torch.optim.Optimizer): The optimizer. Built at the start of `train`, unless `OPTIMIZER` is
            "custom", in which case it must be assigned by the user.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler, if any (see `add_scheduler`).
        callbacks (list[Callback]): Callbacks called at different steps of the training loop.
    """

    def __init__(self,
        config : TrainingConfig
    ):
        """
        Args:
            config (TrainingConfig): The trainer parameters as a configuration object.
                If None, the default `TrainingConfig()` is used.
        """
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
        self._has_scheduler : bool = False
        self.scheduler = None
        self.callbacks = []
        self.metrics = dict()

    def set_training_data(self, data: TensorDataset, shuffle: bool = True):
        """Sets the training dataset. Builds a data loader using `config.BATCH_SIZE` and `config.NUM_DATALOADER_WORKERS`.

        Args:
            data (TensorDataset): The training data.
            shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.

        Raises:
            Exception: if `data` is not a `torch.utils.data.TensorDataset`.
        """
        if not isinstance(data, TensorDataset):
            raise Exception("Please provide a torch.utils.data.TensorDataset object to this function")
        num_workers = self.config.NUM_DATALOADER_WORKERS 
        if num_workers == -1: num_workers = torch.get_num_threads()
        self.train_data_loader = DataLoader(data, batch_size=self.config.BATCH_SIZE, shuffle=shuffle, num_workers=num_workers)

    def set_test_data(self, data: TensorDataset):
        """Sets the test dataset, used by `evaluate_model` at the end of each epoch.
        Builds a data loader using `config.TEST_BATCH_SIZE` and `config.NUM_DATALOADER_WORKERS`.

        Args:
            data (TensorDataset): The test data.

        Raises:
            Exception: if `data` is not a `torch.utils.data.TensorDataset`.
        """
        if not isinstance(data, TensorDataset):
            raise Exception("Please provide a torch.utils.data.TensorDataset object to this function")
        num_workers = self.config.NUM_DATALOADER_WORKERS 
        if num_workers == -1: num_workers = torch.get_num_threads()
        self.test_data_loader = DataLoader(data, batch_size=self.config.TEST_BATCH_SIZE, num_workers=num_workers)

    def get_optimizer(self, model):
        """Builds the optimizer described by `config.OPTIMIZER`, with learning rate `config.LEARNING_RATE`:

        - "adam": `torch.optim.Adam`. Also used for unknown names.
        - "sgd": `torch.optim.SGD` with momentum 0.9.
        - "muon": Muon on weights of dimension >= 2, and Adam (lr 5e-4) on biases and gains. Weight decay 0.01.
        - "custom": no optimizer is built.

        Args:
            model (torch.nn.Module): The model whose parameters are optimized.

        Returns:
            torch.optim.Optimizer: The optimizer, or None if `config.OPTIMIZER` is "custom".
        """
        match self.config.OPTIMIZER.lower():
            case "custom":
                return
            case "sgd":
                return torch.optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE, momentum=0.9)
            case "muon":
                hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
                hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
                param_groups = [
                    dict(params=hidden_weights, use_muon=True,
                        lr=self.config.LEARNING_RATE, weight_decay=0.01),
                    dict(params=hidden_gains_biases, use_muon=False,
                        lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01),
                ]
                return SingleDeviceMuonWithAuxAdam(param_groups)
            case "adam":
                return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            case _:
                # Adam by default
                return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE) 

    def add_scheduler(self, scheduler_cls : torch.optim.lr_scheduler.LRScheduler, *args, **kwargs):
        """Registers a learning rate scheduler. It is instantiated at the start of `train` as
        `scheduler_cls(optimizer, *args, **kwargs)` and stepped once at the end of each epoch.

        Args:
            scheduler_cls (type): The scheduler class (not an instance), for instance `torch.optim.lr_scheduler.StepLR`.
            *args: Positional arguments given to the scheduler after the optimizer.
            **kwargs: Keyword arguments given to the scheduler.

        Example:
            ```python
            trainer.add_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.5)
            ```
        """
        self._scheduler_params = (scheduler_cls, args, kwargs)
        self._has_scheduler = True

    def step_scheduler(self):
        """Steps the learning rate scheduler, if any. Prints the new learning rate when it changes."""
        if self.scheduler is None: return
        last_lr = self.scheduler.get_last_lr()
        self.scheduler.step()
        cur_lr = self.scheduler.get_last_lr()
        if last_lr != cur_lr:
            print("Update learning rate to", cur_lr)

    def add_callbacks(self, *args):
        """Adds callbacks to the trainer. All added callbacks are invoked at the beginning and the end of each training epoch.

        Args:
            *args (Callback): The callbacks, given either as several arguments or as a single iterable.

        Raises:
            AssertionError: if one of the arguments is not a `Callback`.
        """
        if len(args)==1 and isinstance(args[0],Iterable): args = args[0]
        for cb in args:
            assert(isinstance(cb, Callback))
            self.callbacks.append(cb)

    def evaluate_model(self, model):
        """Evaluates the model on the test dataset. Does nothing if no test data was provided.

        Sums the losses returned by the `forward_test_batch` method over all test batches, then calls the `callOnEndTest` method of the callbacks.

        Args:
            model (torch.nn.Module): The model to evaluate.
        """
        if self.test_data_loader is None: return
        test_loss = 0.
        for test_batch in self.test_data_loader:
            test_batch = [_d.to(self.config.DEVICE) for _d in test_batch]
            batch_loss = self.forward_test_batch(test_batch, model)
            test_loss += batch_loss.item()
        self.metrics["test_loss"] = test_loss
        for cb in self.callbacks: cb.callOnEndTest(self, model)

    @abstractmethod
    def forward_test_batch(self, data, model):
        """Computes the test loss over one batch. To be implemented by subclasses.

        Args:
            data (list[torch.Tensor]): The tensors of one batch of the test dataset, already on `config.DEVICE`.
            model (torch.nn.Module): The model being evaluated.

        Returns:
            torch.Tensor: The scalar loss of the batch.
        """
        pass

    @abstractmethod
    def forward_train_batch(self, data, model):
        """Computes the training loss over one batch. To be implemented by subclasses.

        Args:
            data (list[torch.Tensor]): The tensors of one batch of the training dataset, already on `config.DEVICE`.
            model (torch.nn.Module): The model being trained.

        Returns:
            torch.Tensor: The scalar loss of the batch, on which `backward()` is called.
        """
        pass

    def train(self, model, starting_epoch : int =0):
        """Runs the optimization loop for `config.N_EPOCHS` epochs.

        If some were added, callbacks are invoked during training:

        - `callOnBeginTrain` before the first epoch;
        - for each epoch: `callOnBeginEpoch`, `callOnEndForward` after each batch, `callOnEndEpoch`,
          then the scheduler is stepped and the model is evaluated on test data (`callOnEndTest`);
        - `callOnEndTrain` after the last epoch.

        Args:
            model (torch.nn.Module): The model to train. It should already be on `config.DEVICE`.
            starting_epoch (int, optional): Offset of the epoch counter. Used to resume a previous training. Defaults to 0.

        Raises:
            Exception: if no training data was provided, or if `config.OPTIMIZER` is "custom" and `optimizer` was not set.
        """
        if self.train_data_loader is None:
            raise Exception("No training data was provided. Call the `set_training_data` before training.")
        
        if self.config.OPTIMIZER.lower()=="custom": 
            if self.optimizer is None:
                raise Exception("Training configuration has OPTIMIZER type set to 'custom', but no optimizer was provided to the trainer.")
        else:
            self.optimizer = self.get_optimizer(model)
        if self._has_scheduler:
            scheduler_cls, scheduler_args, scheduler_kwargs = self._scheduler_params
            self.scheduler = scheduler_cls(self.optimizer, *scheduler_args, **scheduler_kwargs)

        for cb in self.callbacks: cb.callOnBeginTrain(self,model)

        for epoch in range(self.config.N_EPOCHS):
            epoch += starting_epoch
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks: cb.callOnBeginEpoch(self, model)
            t0 = time.time()
            train_loss = 0. # accumulated loss function over all batches for monitoring purposes
            for data in tqdm(self.train_data_loader, total=len(self.train_data_loader)):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                data = [_d.to(self.config.DEVICE) for _d in data]
                train_batch_loss = self.forward_train_batch(data, model)
                train_batch_loss.backward()
                train_loss += float(train_batch_loss.detach())
                self.optimizer.step()
                for cb in self.callbacks: cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks: cb.callOnEndEpoch(self, model)
            self.step_scheduler()                
            self.evaluate_model(model)

        for cb in self.callbacks: cb.callOnEndTrain(self, model)