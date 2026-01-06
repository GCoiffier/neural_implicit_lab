import torch
from torch.utils.data import DataLoader
import numpy as np

def get_device(force_cpu:bool = False) -> torch.device:
    """Utility function for selecting the correct pytorch device. Returns the first gpu device found on the current system if it exists and "cpu" otherwise.

    Args:
        force_cpu (bool, optional): Forces the computation to happen on the cpu. If set to False, the device returned will always be torch.device("cpu"). Defaults to False.

    Returns:
        torch.device: a torch device on which to run the program.
    """
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda")
    
def forward_in_batches(
        model : torch.nn.Module, 
        inputs : np.ndarray, 
        device : str, 
        compute_grad : bool = False, 
        batch_size : int = 5_000) -> np.ndarray:
    """
    Runs the model `model` onto the `inputs` array and returns the computed output. As the input array may be large, the function splits it in batches that are fed one after the other to the model.

    Args:
        model (torch.nn.Module): the pytorch model to run
        inputs (numpy.ndarray|torch.Tensor): Input array to be fed to the model 
        device (str): which device the computation should be run on.
        compute_grad (bool, optional): whether to also compute and returns the gradients of the model at the input points. Defaults to False.
        batch_size (int, optional): Batch size for the forward computation. Defaults to 5_000.

    Returns:
        numpy.ndarray: the output `model(inputs)`. If `compute_grad` is set to True, also returns the gradients as another `numpy.ndarray` 
    """
    inputs = torch.Tensor(inputs).to(device)
    inputs = DataLoader(inputs, batch_size=batch_size)
    outputs = []
    grads = []
    for batch in inputs:
        batch.requires_grad = compute_grad
        v_batch = model(batch)
        if compute_grad:
            torch.sum(v_batch).backward()
            grads.append(batch.grad.detach().cpu().numpy())
        outputs.append(v_batch.detach().cpu().numpy())
    
    if compute_grad:
        return np.concatenate(outputs), np.concatenate(grads)
    else:
        return np.concatenate(outputs)
    

def gradient(inp_tensor: torch.Tensor, out_tensor: torch.Tensor) -> torch.Tensor:
    """Computes the gradient of the output tensor with respect to the input tensor using pytorch's autograd.

    Args:
        inp_tensor (torch.Tensor): input tensor
        out_tensor (torch.Tensor): output tensor

    Returns:
        torch.Tensor: the gradient
    """
    assert inp_tensor.requires_grad
    grad_outputs = torch.ones_like(out_tensor, device=inp_tensor.device)
    grad = torch.autograd.grad(out_tensor, inp_tensor, 
        grad_outputs=grad_outputs, 
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return grad