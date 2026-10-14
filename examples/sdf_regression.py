import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab as IL
from implicitlab.training import TrainingConfig
from implicitlab.training import callbacks

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

field = IL.fields.Distance(geometry, signed=True, square=False)
sampling_strat = IL.sampling_strategies.CombinedStrategy([
    IL.sampling_strategies.UniformBox(geometry),
    IL.sampling_strategies.NearGeometryGaussian(geometry)
], [1., 1.])

sampler = IL.PointSampler(geometry, sampling_strat, field)
points, val = sampler.sample(2**20, on_ratio=0.2)
train_data = IL.data.make_tensor_dataset((points, val))

# test_pts, test_val = sampler.sample(2**14, on_ratio=0.1)
# test_data = IL.data.make_tensor_dataset((test_pts, test_val))

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

###### Training 

# setup model
# model = IL.nn.MultiLayerPerceptron(geometry.dim, 128, 8).to(DEVICE)
# model = IL.nn.DenseLipSDP(geometry.dim, 128, 6).to(DEVICE)
model = IL.nn.SirenNet(geometry.dim, 128, 5, init_type="winner").to(DEVICE)
# model = IL.nn.QuaNet(geometry.dim, dim_hidden=64, n_layers=4, activation=IL.nn.siren.SinusActivation).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE = 2**15,
    TEST_BATCH_SIZE = 2**16,
    N_EPOCHS = 1000,
    LEARNING_RATE = 1e-4,
    DEVICE = DEVICE,
    OPTIMIZER = "muon",
    COMPILE=True,
    MATMUL_PRECISION="high"
)

trainer = IL.training.SimpleRegressionTrainer(config)
# trainer = IL.training.RegressionEikonalTrainer(config, eikonal_weight=1e-4)

trainer.add_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, trainer.config.N_EPOCHS, eta_min=1e-7)
trainer.add_callbacks(
    callbacks.PrinterCB(), 
    callbacks.ResampleCB(sampler, 2**20, on_ratio=0.2, freq=trainer.config.N_EPOCHS//2)
)
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, 100, plot_domain=M.geometry.AABB([-1.2, -1.2], [1.2, 1.2]), n_contours=31))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 100, res=256, iso=0.))
trainer.set_training_data(train_data)
# trainer.set_test_data(test_data)

trainer.train(model)
model.save(os.path.join(OUTPUT_DIR, "model.inr"))