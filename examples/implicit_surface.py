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
argument_parser.add_argument("-nl", "--n-layers", type=int, default=5, help="Number of layers in the neural network")
argument_parser.add_argument("-ls", "--layer-size", type=int, default=128, help="size of each layer in the neural network")
argument_parser.add_argument("-np", "--n-points", type=int, default=2**18, help="Number of sampled point in the training dataset")
argument_parser.add_argument("-ne", type=int, default=1000, help="Number training epochs")
args = argument_parser.parse_args()

np.random.seed(42)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(args.input_geometry)
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

points, normals = IL.data.sample_geometry_with_normals(geometry, args.n_points)
train_data = IL.data.make_tensor_dataset((points, normals))

###### Training 

# Setup model
model = IL.nn.SirenNet(geometry.dim, args.layer_size, args.n_layers, init_type="siren").to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
trainer = ImplicitSurfaceTrainer(TrainingConfig(
    BATCH_SIZE=2**13,
    TEST_BATCH_SIZE = 2**17,
    N_EPOCHS=args.ne,
    LEARNING_RATE = 1e-3,
    OPTIMIZER="adam",
    DEVICE=DEVICE,
    MATMUL_PRECISION="high",
    COMPILE= False # there are differential losses involved, compile is not supported
))

trainer.add_callbacks(callbacks.PrinterCB())
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, 100))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 100, res=256, iso=0.))
trainer.set_training_data(train_data)
trainer.add_scheduler(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, args.ne//5, eta_min=1e-7)
trainer.train(model)

model.save(os.path.join(OUTPUT_DIR, "model.inr"))