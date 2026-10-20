import implicitlab as IL
import torch
import numpy as np
import os

from data.test_data import *

def test_save_and_load():
    model = IL.nn.ImplicitSequential(
        IL.nn.encodings.RandomFourierEncoding(2, 64),
        IL.nn.MultiLayerPerceptron(64, 128, 3)
    )
    inputs = torch.rand((1000,2))
    with torch.no_grad():
        out1 = model(inputs).numpy()
        model.save("_test_model.inr")
        model = IL.load_model("_test_model.inr", "cpu")
        out2 = model(inputs).detach().numpy()
        assert np.linalg.norm(out1-out2)<1e-14
        try:
            os.remove("_test_model.inr")
        except FileNotFoundError:
            pass