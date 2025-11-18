import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def make_tensor_dataset(data:list, device:str) -> TensorDataset:
    """Aggregates a list of numpy arrays into a pytorch TensorDataset object for training

    Args:
        data (list): list of numpy.array object
        device (str): the device onto which the dataset is loaded. Common choices are 'cpu' for cpu training and "cuda" for GPU training

    Returns:
        TensorDataset: the formatted dataset
    """
    tensors = []
    for tensor in data:
        if len(tensor.shape)==1:
            tensor = tensor.reshape((-1,1))
        tensors.append(torch.Tensor(tensor).to(device))
    return TensorDataset(*tensors)


def load_dataset(file_path:str, device:str) -> TensorDataset:
    raise NotImplementedError
    return


def save_dataset(data: TensorDataset, file_path:str):
    raise NotImplementedError
    return