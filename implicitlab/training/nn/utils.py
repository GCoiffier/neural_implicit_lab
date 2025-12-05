import torch

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of optimizable parameters inside a pytorch neural network.

    Args:
        model (torch.nn.Module): the neural model to consider

    Returns:
        int: number of optimizable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
