import os, sys
import mouette as M
import numpy as np

import implicitlab as IL
from implicitlab.data import PointSampler
from implicitlab.training import TrainingConfig, hKRTrainer, callbacks

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()
print("DEVICE:", DEVICE)

####### Dataset Sampling

# training data
train_field = IL.fields.Occupancy(geometry, v_in=-1, v_out=1, v_on=-1)
train_sampling_strat = IL.sampling_strategies.CombinedStrategy([
    IL.sampling_strategies.UniformBox(geometry),
    IL.sampling_strategies.NearGeometryGaussian(geometry, 0.02)
], [1., 1.])
# train_sampling_strat = IL.sampling_strategies.UniformBox(geometry)
train_sampler = PointSampler(geometry, train_sampling_strat, train_field)
points, val = train_sampler.sample(2**20, on_ratio=0.)

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
train_data = IL.data.make_tensor_dataset((points, val)) 

pc = M.mesh.from_arrays(points)
pc.vertices.register_array_as_attribute("occ", val)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))
M.mesh.save(geometry, os.path.join(OUTPUT_DIR, "input_geometry.obj"))

# testing data
# test_field = IL.fields.Distance(geometry, signed=True)
# test_sampling_strat = IL.sampling_strategies.UniformBox(geometry)
# test_sampler = PointSampler(geometry, test_sampling_strat, test_field)
# test_pts, test_val = test_sampler.sample(2**13)
# test_data = IL.data.make_tensor_dataset((test_pts, test_val))


###### Training
if geometry.dim == 2:
    n_layers, layer_size = 10, 64
elif geometry.dim == 3:
    n_layers, layer_size = 12, 128


# model = IL.nn.DenseLipBjorck(geometry.dim, layer_size, n_layers).to(DEVICE)
# model = IL.nn.DenseLipAOL(geometry.dim, layer_size, n_layers2).to(DEVICE)
# model = IL.nn.DenseLipCPL(geometry.dim, layer_size, n_layers).to(DEVICE)
model = IL.nn.DenseLipSDP(geometry.dim, layer_size, n_layers).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")


# Setup trainer
class UpdateHkrRegulCB(callbacks.Callback):
    def __init__(self, when : dict):
        super().__init__()
        self.when = when

    def callOnBeginEpoch(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            trainer.lossfun.lmbd = self.when[epoch]
            print("Updated loss regul weight to", self.when[epoch])


MARGIN = 0.01

trainer = hKRTrainer(TrainingConfig(
    BATCH_SIZE=2**15,
    TEST_BATCH_SIZE=2**17,
    N_EPOCHS=3000,
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE,
    OPTIMIZER="adam",
    MATMUL_PRECISION="high",
    COMPILE=True
    ),
    margin=MARGIN, 
    lmbd=100.
)

trainer.add_callbacks(
    callbacks.PrinterCB(),
    # callbacks.LoggerCB(os.path.join(OUTPUT_DIR, "training_log.txt")),
    # callbacks.CheckpointCB("output", [x for x in range(config.N_EPOCHS) if x%50==0]),
    # UpdateHkrRegulCB({500: 100., 2000: 1000.})
)

if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, 500, n_contours=31))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 500, res=256, iso=0))
# trainer.add_scheduler(CosineAnnealingLR, trainer.config.N_EPOCHS, 1e-3*trainer.config.LEARNING_RATE)

trainer.set_training_data(train_data)
# trainer.set_test_data(test_data)
trainer.train(model)

model.save(os.path.join(OUTPUT_DIR, "model.inr"))
