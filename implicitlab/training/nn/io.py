import torch

def save_model(model: torch.nn.Module, path: str):
    """Saves a neural model onto the disk.

    Args:
        model (torch.nn.Module): neural model to be saved
        path (str): file path to the saved file.
    """
    data = { "model": model, "state_dict" : model.state_dict()}
    torch.save(data, path)
    
def load_model(path:str, device:str = "cpu") -> torch.nn.Module:
    """Loads a neural model from the disk

    Args:
        path (str): file path to the saved file
        device (str, optional): device onto which the model is loaded. Defaults to "cpu".

    Returns:
        torch.nn.Module: the loaded neural model
    """
    data = torch.load(path, map_location=device, weights_only=False)
    model = data["model"]
    model.load_state_dict(data["state_dict"])
    return model.to(device)