import os, sys
import mouette as M
import numpy as np
import torch
from torch import nn

import implicitlab as IL
from implicitlab.training import TrainingConfig
from implicitlab.training import callbacks


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry)
print(geometry.geom_type)
DEVICE = IL.utils.get_device()
HASH = True
DOMAIN = M.geometry.AABB([-1.5]*geometry.dim, [1.5]*geometry.dim)


field = IL.fields.Distance(geometry, signed=True, square=False)
sampling_strat = IL.sampling_strategies.CombinedStrategy([
    IL.sampling_strategies.UniformBox(geometry),
    IL.sampling_strategies.NearGeometryGaussian(geometry)
], [1., 1.])
sampler = IL.PointSampler(geometry, sampling_strat, field)
points, val = sampler.sample(2**20, on_ratio=0.1)
train_data = IL.data.make_tensor_dataset((points, val))

if HASH:
    enc = IL.nn.encodings.MultiResHashGridEncoding(geometry.dim, DOMAIN)
else:
    enc = IL.nn.encodings.MultiResDenseGridEncoding(geometry.dim, DOMAIN)

model = IL.nn.ImplicitSequential(enc, IL.nn.MultiLayerPerceptron(enc.output_dim,64,3)).to(DEVICE)
print(f"Parameter count: {IL.nn.count_parameters(model)}")

# Setup trainer
trainer = IL.training.SimpleRegressionTrainer(TrainingConfig(
    BATCH_SIZE = 2**15,
    TEST_BATCH_SIZE = 2**16,
    N_EPOCHS = 100,
    LEARNING_RATE = 1e-2,
    DEVICE = DEVICE,
    OPTIMIZER = "custom",
    COMPILE=True
))

trainer.optimizer = torch.optim.Adam([
    {"params": model[0].parameters(), "weight_decay": 0.}, # as in instant-ngp: no weight decay on the hash table entries
    {"params": model[1].parameters(), "weight_decay": 1e-6},
], trainer.config.LEARNING_RATE, betas=(0.9, 0.99), eps=1e-15)

trainer.add_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, trainer.config.N_EPOCHS, 1e-8)
trainer.add_callbacks(callbacks.PrinterCB())

if geometry.dim==2:
    trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, 100))
elif geometry.dim==3:
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 500, res=512, iso=0, full_precision=True))

trainer.set_training_data(train_data)
trainer.train(model)
model.save(os.path.join(OUTPUT_DIR, "model.inr"))
