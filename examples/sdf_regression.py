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
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

field = IL.fields.Distance(geometry, signed=True, square=False)
# field = implicitlab.fields.Occupancy(geometry, v_in=0, v_out=1, v_on=0)
# field = implicitlab.fields.CustomFunction(lambda p : np.cos(5*p[0])*np.cos(5*p[1]))

# sampling_strat = implicitlab.sampling_strategy.UniformBox(geometry)
# sampling_strat = IL.sampling_strategy.NearGeometryGaussian(geometry)
sampling_strat = IL.sampling_strategy.CombinedStrategy([
    IL.sampling_strategy.UniformBox(geometry),
    IL.sampling_strategy.NearGeometryGaussian(geometry)
], [2., 1.])


sampler = IL.PointSampler(geometry, sampling_strat, field)
points, val = sampler.sample(200_000, on_ratio=0.2)
train_data = IL.data.make_tensor_dataset((points, val), DEVICE)

test_pts, test_val = sampler.sample(10_000, on_ratio=0.1)
test_data = IL.data.make_tensor_dataset((test_pts, test_val), DEVICE)

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("val", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))

###### Training 

# setup model
# model = IL.nn.MultiLayerPerceptron(geometry.dim, 128, 10).to(DEVICE)
# model = IL.nn.DenseLipSDP(geometry.dim, 128, 6).to(DEVICE)
model = IL.nn.SirenNet(geometry.dim, 128, 6).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
config = TrainingConfig(
    BATCH_SIZE=100,
    TEST_BATCH_SIZE = 10000,
    N_EPOCHS=100,
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE,
    OPTIMIZER="adam"
)

# trainer = IL.training.SimpleRegressionTrainer(config, lossfun=torch.nn.MSELoss())
trainer = IL.training.RegressionEikonalTrainer(config, eikonal_weight=1e-3)

trainer.add_callbacks(callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, 10, plot_domain=M.geometry.AABB([-1.2, -1.2], [1.2, 1.2])))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 50, res=400, iso=[0., 0.1]))

trainer.set_training_data(train_data)
trainer.set_test_data(test_data)
trainer.train(model)

IL.nn.save_model(model, os.path.join(OUTPUT_DIR, "model.pt"))
