import os, sys
import mouette as M
import torch
import argparse
import numpy as np

import implicitlab as IL
from implicitlab.training.losses import EikonalLoss
from implicitlab.training import TrainingConfig,Trainer
from implicitlab.training import callbacks


argument_parser = argparse.ArgumentParser(
    prog="Neural Implicit Surface",
    description="Fitting a SirenNet to an implicit surface"
)
argument_parser.add_argument("input_geometry", type=str)
argument_parser.add_argument("-o", "--output-name", default="", type=str)
argument_parser.add_argument("-nl", "--n-layers", type=int, default=6, help="Number of layers in the neural network")
argument_parser.add_argument("-ls", "--layer-size", type=int, default=400, help="size of each layer in the neural network")
argument_parser.add_argument("-np", "--n-points", type=int, default=500_000, help="Number of sampled point in the training dataset")
argument_parser.add_argument("-ne", type=int, default=150, help="Number training epochs")
args = argument_parser.parse_args()

np.random.seed(42)

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(args.input_geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

if geometry.dim == 3:
    points, normals = M.sampling.sample_surface(geometry, args.n_points, return_normals=True)
elif geometry.dim == 2:
    points, normals = IL.data.sample_points_and_normals2D(geometry, 50_000)
train_data = IL.data.make_tensor_dataset((points, normals), DEVICE)

pc = M.mesh.from_arrays(points)
M.mesh.save(pc, "output/train_pts.geogram_ascii")

###### Training 

# Setup model
model = IL.nn.SirenNet(geometry.dim, args.layer_size, args.n_layers).to(DEVICE)

# model = torch.nn.Sequential(
#     # IL.nn.encodings.HalfPlaneEncoding(geometry, 1000),
#     # IL.nn.encodings.PointDistanceEncoding(geometry, 1000),
#     IL.nn.encodings.RandomFourierEncoding(geometry, 1000),
#     # IL.nn.encodings.GaussianEncoding(geometry, 1000),
#     IL.nn.MultiLayerPerceptron(1000, 256, 10)
# ).to(DEVICE)

print(f"{IL.nn.count_parameters(model)} parameters")

if args.output_name:
    torch.save(model.state_dict(), f"output/{args.output_name}.pt")
else:
    torch.save(model.state_dict(), f"output/siren_{args.layer_size}x{args.n_layers}.pt")

# Setup trainer
class ImplicitSurfaceTrainer(Trainer):

    def __init__(self, 
        config : TrainingConfig
    ):
        super().__init__(config)
        self.rho = 100.
        self.weights = {
            "eikonal" : 50.,
            "on" : 7000.,
            "out" : 600.,
            "normals": 100.,
        }
    
    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
    
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


trainer = ImplicitSurfaceTrainer(TrainingConfig(
    BATCH_SIZE=1000,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=args.ne,
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE
))

trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 10))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 50, res=400, iso=0.))
trainer.set_training_data(train_data)
trainer.train(model)

if args.output_name:
    torch.save(model.state_dict(), f"output/{args.output_name}.pt")
else:
    torch.save(model.state_dict(), f"output/siren_{args.layer_size}x{args.n_layers}.pt")