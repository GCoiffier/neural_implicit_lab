import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, NeuralPullTrainer
from implicitlab.training import callbacks


os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

sampler = IL.PointSampler(
    geometry, 
    IL.sampling_strategy.NearGeometryGaussian(geometry, stdv=0.3), 
    IL.fields.Nearest(geometry)
)

points, val = sampler.sample(500_000, on_ratio=0)
train_data = IL.data.make_tensor_dataset((points, val), DEVICE)
visu_vector_field = M.procedural.vector_field(points, val-points)
M.mesh.save(visu_vector_field, "output/project.mesh")

# test_pts, test_val = sampler.sample(10_000)
# test_data = IL.data.make_tensor_dataset((test_pts, test_val), DEVICE)

###### Training 

# setup model
model = IL.nn.MultiLayerPerceptron(geometry.dim, 128, 6).to(DEVICE)
# model = IL.nn.SirenNet(geometry.dim, 128, 4).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=5000,
    N_EPOCHS=200,
    DEVICE=DEVICE
)   

trainer = NeuralPullTrainer(config)

trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"),
    callbacks.Render2DCB("output", 10),
    # callbacks.CheckpointCB("output", [x+1 for x in range(config.N_EPOCHS-1) if x%10==0])
)
trainer.set_training_data(train_data)
# trainer.set_test_data(test_data)
trainer.train(model)
IL.nn.save_model(model, "output/model.pt")