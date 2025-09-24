import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deel import torchlip

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    data = { "id": model.id, "meta" : model.meta, "state_dict" : model.state_dict()}
    torch.save(data, path)
    

def load_model(path, device:str):
    data = torch.load(path, map_location=device)
    model_type = data.get("id","Spectral")
    if model_type == "SDP":
        model = DenseSDP(*data["meta"])
    else:
        raise Exception(f"Model type {model_type} not recognized")
    model.load_state_dict(data["state_dict"])
    return model.to(device)


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class SDPBasedLipschitzDense(nn.Module):

    def __init__(self, in_features, inner_dim=-1):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = nn.ReLU()

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out
    
def DenseSDP(dim_in, dim_hidden, n_layers):
    layers = []
    layers.append(nn.ZeroPad1d((0, dim_hidden-dim_in)))
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(dim_hidden))
    layers.append(torchlip.FrobeniusLinear(dim_hidden,1))
    model = torch.nn.Sequential(*layers)
    model.id = "SDP"
    model.meta = [dim_in, dim_hidden, n_layers]
    return model