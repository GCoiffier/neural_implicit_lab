import os, sys
import mouette as M
import torch
import argparse
import numpy as np

import implicitlab as IL
from implicitlab.training import TrainingConfig, ImplicitSurfaceTrainer
from implicitlab.training import callbacks


argument_parser = argparse.ArgumentParser(
    prog="Neural Implicit Surface",
    description="Fitting a SirenNet to an implicit surface"
)
argument_parser.add_argument("input_geometry", type=str)
argument_parser.add_argument("-o", "--output-name", default="", type=str)
argument_parser.add_argument("-nl", "--n-layers", type=int, default=6, help="Number of layers in the neural network")
argument_parser.add_argument("-ls", "--layer-size", type=int, default=128, help="size of each layer in the neural network")
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
    points, normals = IL.data.sample_points_and_normals2D(geometry, 100_000)
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
trainer = ImplicitSurfaceTrainer(TrainingConfig(
    BATCH_SIZE=1000,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=args.ne,
    LEARNING_RATE=1e-4,
    OPTIMIZER="Adam",
    DEVICE=DEVICE
))

trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 10))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 50, res=400, iso=[0., 0.1]))
trainer.set_training_data(train_data)
trainer.train(model)

if args.output_name:
    torch.save(model.state_dict(), f"output/{args.output_name}.pt")
else:
    torch.save(model.state_dict(), f"output/siren_{args.layer_size}x{args.n_layers}.pt")