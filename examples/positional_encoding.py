import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab as IL
from implicitlab.training import TrainingConfig
from implicitlab.training import callbacks

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

field = IL.fields.Distance(geometry, signed=True, square=False)
sampling_strat = IL.sampling_strategy.CombinedStrategy([
    IL.sampling_strategy.UniformBox(geometry),
    IL.sampling_strategy.NearGeometryGaussian(geometry)
], [2., 1.])

sampler = IL.PointSampler(geometry, sampling_strat, field)
points, val = sampler.sample(100_000)
train_data = IL.data.make_tensor_dataset((points, val), DEVICE)

test_pts, test_val = sampler.sample(5_000)
test_data = IL.data.make_tensor_dataset((test_pts, test_val), DEVICE)

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, "output/train_pts.geogram_ascii")

###### Training 

# setup model
model = torch.nn.Sequential(
    # IL.nn.encodings.HalfPlaneEncoding(geometry, 1000),
    # IL.nn.encodings.PointDistanceEncoding(geometry, 1000),
    # IL.nn.encodings.RandomFourierEncoding(geometry, 1000),
    IL.nn.encodings.GaussianEncoding(geometry, 1000),
    IL.nn.MultiLayerPerceptron(1000, 128, 5)
).to(DEVICE)
# model = IL.nn.MultiLayerPerceptron(geometry.dim, 128, 5).to(DEVICE)

print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=100,
    TEST_BATCH_SIZE = 10000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE
)

trainer = IL.training.SimpleRegressionTrainer(config, lossfun=torch.nn.MSELoss())

trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"),
)
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 10))
elif geometry.dim == 3:
    domain = M.geometry.AABB([-1.5]*geometry.dim, [1.5]*geometry.dim)
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 10, res=400))

trainer.set_training_data(train_data)
trainer.set_test_data(test_data)
trainer.train(model)

IL.nn.save_model(model, "output/model.pt")