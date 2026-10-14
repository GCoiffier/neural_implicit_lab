import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

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


def get_model_dim(model: torch.nn.Module, device):
    try:
        rdm_data = torch.rand((10,2)).to(device)
        _ = model(rdm_data)
        return 2
    except:
        rdm_data = torch.rand((10,3)).to(device)
        _ = model(rdm_data)
        return 3

def forward_in_batches(
        model : torch.nn.Module, 
        inputs : torch.Tensor, 
        device : torch.device, 
        compute_grad : bool = False, 
        batch_size : int = 2**16,
        use_tqdm : bool = False,
        full_precision : bool = True
    ) -> torch.Tensor:
    """
    Runs the model `model` onto the `inputs` array and returns the computed output. As the input array may be large, the function splits it in batches that are fed one after the other to the model.

    Args:
        model (torch.nn.Module): the pytorch model to run
        inputs (torch.Tensor): Input array to be fed to the model. Also supports numpy array, which will be converted automatically.
        device (torch.device): which device the computation should be run on.
        compute_grad (bool, optional): whether to also compute and returns the gradients of the model at the input points. Defaults to False.
        batch_size (int, optional): Batch size for the forward computation. Defaults to 2**16.
        use_tqdm (bool, optional): Whether to display a tqdm progress bar during computation. Defaults to False.
        full_precision (bool, optional): Whether to compute with torch.float32 (more accurate) or torch.float16 (faster but some slight errors may occur). Defaults to True.

    Returns:
        torch.Tensor: the output `model(inputs)`. If `compute_grad` is set to True, also returns the gradients as another tensor.
    """
    inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    n_inputs = inputs.shape[0]
    batch_starts = range(0, n_inputs, batch_size)
    if use_tqdm: batch_starts = tqdm(batch_starts, total=len(batch_starts))
    outputs = None # will be created on the fly since its shape is unknown at first
    grads = torch.empty_like(inputs) if compute_grad else None
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.set_grad_enabled(compute_grad), torch.autocast(device.type, dtype=amp_dtype, enabled= not full_precision):
        for i in batch_starts:
            batch = inputs[i:i+batch_size]
            if compute_grad:
                batch.requires_grad = True
            v_batch = model(batch)
            if compute_grad:
                grads[i:i+batch_size] = torch.autograd.grad(v_batch.sum(), batch)[0]
                v_batch = v_batch.detach()
            if outputs is None:
                outputs = torch.empty((n_inputs, *v_batch.shape[1:]), dtype=torch.float32, device=device)
            outputs[i:i+batch_size] = v_batch
    return (outputs, grads) if compute_grad else outputs