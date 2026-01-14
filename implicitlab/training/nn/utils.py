import torch

def count_parameters(model: torch.nn.Module, with_grad:bool = True) -> int:
    """Counts the number of optimizable parameters inside a pytorch neural network.

    Args:
        model (torch.nn.Module): the neural model to consider
        with_grad (bool, optional): whether to count all the parameters in the model's tensors, or only optimizable parameters (with requires_grad=True). Defaults to True.

    Returns:
        int: number of optimizable parameters in the model
    """
    if with_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())