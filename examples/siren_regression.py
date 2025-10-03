import os, sys
import mouette as M
import numpy as np
import torch

import implicitlab
from implicitlab.training import SimpleRegressionTrainer, RegressionEikonalTrainer, TrainingConfig
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

sampler = implicitlab.PointSampler(geometry, sampling_strat, field)
points, val = sampler.sample(80_000)
train_data = implicitlab.data.make_tensor_dataset((points, val), DEVICE) 
test_pts, test_val = sampler.sample(5000)
test_data = implicitlab.data.make_tensor_dataset((test_pts, test_val), DEVICE)

###### Training 

# setup model
# model = implicitlab.nn.MultiLayerPerceptron(geometry.dim, 128, 8).to(DEVICE)
model = implicitlab.nn.SirenNet(geometry.dim, 64, 5).to(DEVICE)
print(f"{implicitlab.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=200,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    DEVICE=DEVICE
)

loss = torch.nn.MSELoss()
# trainer = SimpleRegressionTrainer(config, lossfun=loss)
trainer = RegressionEikonalTrainer(config, eikonal_weight=1e-5)
trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"),
    callbacks.Render2DCB("output", 50),
    callbacks.CheckpointCB("output", [x+1 for x in range(config.N_EPOCHS-1) if x%50==0])
)
trainer.set_training_data(train_data)
trainer.set_test_data(test_data)
trainer.train(model)

implicitlab.nn.save_model(model, "output/model.pt")