import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab as IL
from implicitlab.training import TrainingConfig, NeuralPullTrainer, callbacks

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

sampler = IL.PointSampler(
    geometry, 
    IL.data.sampling_strategies.NearGeometryGaussian(geometry, stdv=0.1), 
    IL.fields.Nearest(geometry)
)

points, val = sampler.sample(2**19, on_ratio=0)
train_data = IL.data.make_tensor_dataset((points, val))

# test_pts, test_val = sampler.sample(2**15)
# test_data = IL.data.make_tensor_dataset((test_pts, test_val))

###### Training 

# setup model
model = IL.nn.MultiLayerPerceptron(geometry.dim, 256, 8).to(DEVICE)
# model = IL.nn.SirenNet(geometry.dim, 128, 4).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=2**14,
    N_EPOCHS=200,
    DEVICE=DEVICE,
    OPTIMIZER="adam",
    LEARNING_RATE=1e-4
)   

trainer = NeuralPullTrainer(config)

trainer.add_callbacks(
    callbacks.PrinterCB(),
    callbacks.Render2DCB(OUTPUT_DIR, 10),
)
trainer.set_training_data(train_data)
trainer.train(model)
model.save(os.path.join(OUTPUT_DIR, "model.inr"))
