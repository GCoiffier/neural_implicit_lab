import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, hKRTrainer
from implicitlab.training import callbacks


os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()
print("DEVICE:", DEVICE)

####### Dataset Sampling

# training data
train_field = IL.fields.Occupancy(geometry, v_in=-1, v_out=1, v_on=-1)
train_sampling_strat = IL.sampling_strategy.CombinedStrategy([
    IL.sampling_strategy.UniformBox(geometry),
    IL.sampling_strategy.NearGeometryGaussian(geometry)
], [1., 1.])
train_sampler = PointSampler(geometry, train_sampling_strat, train_field)
points, val = train_sampler.sample(100_000)
train_data = IL.data.make_tensor_dataset((points, val), DEVICE) 


# testing data
test_field = IL.fields.Distance(geometry, signed=True)
test_sampling_strat = IL.sampling_strategy.UniformBox(geometry)
test_sampler = PointSampler(geometry, test_sampling_strat, test_field)
test_pts, test_val = test_sampler.sample(10_000)
test_data = IL.data.make_tensor_dataset((test_pts, test_val), DEVICE)


###### Training 
model = IL.nn.DenseSDP(geometry.dim, 100, 10).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=200,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE
)

trainer = hKRTrainer(config, 0.01, 100.)
trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"),
    callbacks.Render2DCB("output", 10),
    callbacks.CheckpointCB("output", [x for x in range(config.N_EPOCHS) if x%50==0])
)
trainer.set_training_data(train_data)
trainer.set_test_data(test_data)
trainer.train(model)
IL.nn.save_model(model, "output/model.pt")