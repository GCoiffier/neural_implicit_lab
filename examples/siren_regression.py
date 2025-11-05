import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab
from implicitlab.training import Trainer, TrainingConfig
from implicitlab.training import callbacks

os.makedirs("output", exist_ok=True)
geometry = implicitlab.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = implicitlab.utils.get_device()

####### Dataset Sampling

field = implicitlab.fields.Distance(geometry, signed=True, square=False)
# field = implicitlab.fields.Occupancy(geometry, v_in=0, v_out=1, v_on=0)
# field = implicitlab.fields.CustomFunction(lambda p : np.cos(5*p[0])*np.cos(5*p[1]))

# sampling_strat = implicitlab.sampling_strategy.UniformBox(geometry)
sampling_strat = implicitlab.sampling_strategy.CombinedStrategy([
    implicitlab.sampling_strategy.UniformBox(geometry),
    implicitlab.sampling_strategy.NearGeometryGaussian(geometry)
], [1., 2.])

K = 1.

sampler = implicitlab.PointSampler(geometry, sampling_strat, field)
points, val = sampler.sample(100_000)
# val = np.clip(K*val, -1, 1)
train_data = implicitlab.data.make_tensor_dataset((points, val), DEVICE)

test_pts, test_val = sampler.sample(10_000)
# test_val = np.clip(K*test_val, -1, 1)
test_data = implicitlab.data.make_tensor_dataset((test_pts, test_val), DEVICE)

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, "output/train_pts.geogram_ascii")

###### Training 

# setup model
# model = implicitlab.nn.MultiLayerPerceptron(geometry.dim, 128, 8).to(DEVICE)
model = implicitlab.nn.SirenNet(geometry.dim, 128, 4).to(DEVICE)
print(f"{implicitlab.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=100,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE
)


class TestTrainer(Trainer):
    def __init__(self, config, lossfun):
        super().__init__(config)
        self.lossfun = lossfun

    def forward_test_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return self.lossfun(Y, Y_target)

    def forward_train_batch(self, data, model):
        X,Y_target = data
        Y = model(X)
        return self.lossfun(Y, Y_target)
    

loss = torch.nn.MSELoss()
trainer = TestTrainer(config, lossfun=loss)
trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"),
    callbacks.Render2DCB("output", 10),
    # callbacks.CheckpointCB("output", [x+1 for x in range(config.N_EPOCHS-1) if x%10==0])
)
trainer.set_training_data(train_data)
trainer.set_test_data(test_data)
trainer.train(model)

implicitlab.nn.save_model(model, "output/model.pt")