import os, sys
import mouette as M
import argparse
import numpy as np

import torch
from torch.nn import functional as F
import implicitlab as IL
from implicitlab.training.losses import EikonalLoss
from implicitlab.training import TrainingConfig,Trainer
from implicitlab.training import callbacks

argument_parser = argparse.ArgumentParser(
    prog="Neural Displacement Field",
    description=""
)

argument_parser.add_argument("input_geometry", type=str)
argument_parser.add_argument("-o", "--output-name", default="", type=str)
argument_parser.add_argument("-nl", "--n-layers", type=int, default=6, help="Number of layers in the neural network")
argument_parser.add_argument("-ls", "--layer-size", type=int, default=128, help="size of each layer in the neural network")
argument_parser.add_argument("-np", "--n-points", type=int, default=200_000, help="Number of sampled point in the training dataset")
argument_parser.add_argument("-ne", "--n-epochs", type=int, default=100, help="Number training epochs")
args = argument_parser.parse_args()

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(args.input_geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling
assert geometry.dim == 3

points, normals = M.sampling.sample_surface(geometry, args.n_points, return_normals=True)
train_data = IL.data.make_tensor_dataset((points, normals), DEVICE)

pc = M.mesh.from_arrays(points)
M.mesh.save(pc, "output/train_pts.geogram_ascii")

###### Training 

# Setup model

def chi(x, nu):
    return 1/(1+torch.pow(x/nu, 4.))

class SirenDisplacementField(torch.nn.Module):

    def __init__(self, layer_size, n_layers):
        super().__init__()
        self.base_siren = IL.nn.SirenNet(3, layer_size, n_layers, w0=15)
        self.detail_siren = IL.nn.SirenNet(3, layer_size, n_layers, w0=60)
        self.kappa = 1.
        self.alpha = 0.05
        self.nu = 0.02

    def forward(self, x):
        x.requires_grad = True
        fx = self.base_siren(x)
        if self.kappa >0 :
            chix = chi(fx, self.nu)
            nx = self.alpha * F.tanh(self.detail_siren(x))
            gd_bx = IL.utils.gradient(x, fx)
            return self.kappa* fx + (1-self.kappa) * self.base_siren(x + chix*nx*F.normalize(gd_bx))
        return fx

model = SirenDisplacementField(args.layer_size, args.n_layers).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

class ImplicitSurfaceTrainer(Trainer):

    def __init__(self, 
        config : TrainingConfig
    ):
        super().__init__(config)
        self.rho = 100.
        self.weights = {
            "eikonal" : 5.,
            "on" : 400.,
            "normals": 40.,
            "out" : 50.,
        }

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-5)
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts, normals = data
        pts.requires_grad = True
        Y_on = model(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        pts_out = 3*torch.rand_like(pts)-1.5
        pts_out.requires_grad = True
        Y_out = model(pts_out)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(Y_out)))

        grad = torch.autograd.grad(Y_on, pts, grad_outputs=torch.ones_like(Y_on), create_graph=True)[0]
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad, normals)
        
        batch_loss += self.weights["eikonal"] * EikonalLoss()(pts_out, Y_out)        
        return batch_loss


class KappaUpdateCallback(callbacks.Callback):
    
    def __init__(self, tm, n_epochs):
        self.tm = tm
        self.n = n_epochs

    def callOnEndEpoch(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        t = epoch/self.n
        if t<self.tm:
            model.kappa = 0.
        else:
            model.kappa = 0.5*(1 + np.cos(np.pi * (t - self.tm)/(1 - self.tm)))


trainer = ImplicitSurfaceTrainer(TrainingConfig(
    BATCH_SIZE = 4096,
    TEST_BATCH_SIZE = 10_000,
    N_EPOCHS=args.n_epochs,
    LEARNING_RATE=5e-5,
    DEVICE=DEVICE
))

TRAIN_PERCENT = 0.2

trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
trainer.add_callbacks(callbacks.MarchingCubeCB("output", int(args.n_epochs*TRAIN_PERCENT), res=400, iso=[0., 0.05]))
trainer.add_callbacks(KappaUpdateCallback(TRAIN_PERCENT, args.n_epochs))
trainer.set_training_data(train_data)
trainer.train(model)

print("KAPPA", model.kappa)
# model.kappa = 0.
# iso_surfaces = IL.visualize.reconstruct_surface_marching_cubes(model, plot_domain, DEVICE, iso=[0., 0.01, 0.05, 0.1], res=400, batch_size=10_000, use_tqdm=True)
# for (n,off),mesh in iso_surfaces.items():
#     M.mesh.save(mesh, os.path.join("output", f"e_final_iso{round(1000*off)}.obj"))

torch.save(model.state_dict(), f"output/idf.pt")