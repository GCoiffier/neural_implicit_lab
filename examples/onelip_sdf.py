import os, sys
import mouette as M
import numpy as np

import torch

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, hKRTrainer, callbacks


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
    IL.sampling_strategy.NearGeometryGaussian(geometry, 0.03)
], [1., 2.])
train_sampler = PointSampler(geometry, train_sampling_strat, train_field)
points, val = train_sampler.sample(10_000 if geometry.dim==2 else 100_000)

# Balance the dataset : as many inside points that there are outside points
points_pos = points[val>0, :]
points_neg = points[val<0, :]
n_pos, n_neg = points_pos.shape[0], points_neg.shape[0]
print(n_pos, "outside points")
print(n_neg, "inside points")
if n_pos<n_neg:
    points_neg = points_neg[:n_pos, :]
elif n_pos>n_neg:
    points_pos = points_pos[:n_neg, :]
points = np.concatenate((points_pos, points_neg))
val = np.concatenate((np.ones(min(n_pos,n_neg)), -np.ones(min(n_pos,n_neg))))
train_data = IL.data.make_tensor_dataset((points, val), DEVICE) 

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("occ", val)
M.mesh.save(pc, "output/train_pts.geogram_ascii")

# testing data
test_field = IL.fields.Distance(geometry, signed=True)
test_sampling_strat = IL.sampling_strategy.UniformBox(geometry)
test_sampler = PointSampler(geometry, test_sampling_strat, test_field)
test_pts, test_val = test_sampler.sample(10_000)
test_data = IL.data.make_tensor_dataset((test_pts, test_val), DEVICE)


###### Training
# model = IL.nn.DenseLipBjorck(geometry.dim, 128, 20).to(DEVICE)
# model = IL.nn.DenseLipAOL(geometry.dim, 128, 10).to(DEVICE)
# model = IL.nn.DenseLipSDP(geometry.dim, 128, 20, activation=nn.Softplus(10)).to(DEVICE)
model = IL.nn.DenseLipSDP(geometry.dim, 128, 20).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=1000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-2,
    DEVICE=DEVICE
)

class UpdateHkrRegulCB(callbacks.Callback):
    def __init__(self, when : dict):
        super().__init__()
        self.when = when

    def callOnBeginTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            trainer.lossfun.lmbd = self.when[epoch]
            print("Updated loss regul weight to", self.when[epoch])


trainer = hKRTrainer(config, 0.01, 100.)
trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"),
    # callbacks.CheckpointCB("output", [x for x in range(config.N_EPOCHS) if x%50==0]),
    UpdateHkrRegulCB({1: 1., 5 : 10., 10 : 100.})
)

if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 10))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 50, iso=[-0.01, 0]))

trainer.set_training_data(train_data)
trainer.set_test_data(test_data)
trainer.train(model)

IL.nn.save_model(model, "output/model.pt")
