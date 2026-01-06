import torch

def save_model(model: torch.nn.Module, path: str):
    """Saves a neural model onto the disk.

    Args:
        model (torch.nn.Module): neural model to be saved
        path (str): file path to the saved file.
    """
    for tnsr in model.parameters(): break
    try:
        rdm_data = torch.rand((10,2)).to(tnsr.device)
        traced_model = torch.jit.trace(model, rdm_data)
    except:
        rdm_data = torch.rand((10,3)).to(tnsr.device)
        traced_model = torch.jit.trace(model, rdm_data)
    traced_model.save(path)
    
def load_model(path:str, device:str = "cpu") -> torch.nn.Module:
    """Loads a neural model from the disk.

    Args:
        path (str): file path to the saved file
        device (str, optional): device onto which the model is loaded. Defaults to "cpu".

    Returns:
        torch.nn.Module: the loaded neural model
    """
    model = torch.jit.load(path, map_location=device)
    try:
        rdm_data = torch.rand((10,2)).to(device)
        _ = model(rdm_data)
        model.dim = 2
    except:
        rdm_data = torch.rand((10,3)).to(device)
        _ = model(rdm_data)
        model.dim = 3
    return model