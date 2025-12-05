import implicitlab as IL
import torch
import numpy as np
import os

from data.test_data import *

def test_save_and_load():
    geometry = dauphin2d_pl()

    model = torch.nn.Sequential(
        IL.nn.encodings.RandomFourierEncoding(geometry, 64),
        IL.nn.MultiLayerPerceptron(64, 128, 3)
    ).to("cpu")

    inp = torch.rand((100,geometry.dim))
    out1 = model(inp).detach().numpy()
    IL.nn.save_model(model, "model.pt")
    model = IL.nn.load_model("model.pt", "cpu")
    out2 = model(inp).detach().numpy()
    assert np.linalg.norm(out1-out2)<1e-14
    os.remove("model.pt")