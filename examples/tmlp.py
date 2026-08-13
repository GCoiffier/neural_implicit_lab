import os
import mouette as M
import argparse
import numpy as np
from tqdm import tqdm
from skimage.measure import marching_cubes

import torch
from torch.utils.data import DataLoader, TensorDataset

import implicitlab as IL
from implicitlab.training import Trainer, TrainingConfig, SphereInitializeTrainer
from implicitlab.training import callbacks


argument_parser = argparse.ArgumentParser(
    prog="Neural Implicit Surface",
    description="Fitting a SirenNet to an implicit surface"
)
argument_parser.add_argument("input_geometry", type=str)
argument_parser.add_argument("-o", "--output-name", default="", type=str)
argument_parser.add_argument("-nl", "--n-layers", type=int, default=7, help="Number of layers in the neural network")
argument_parser.add_argument("-ls", "--layer-size", type=int, default=256, help="size of each layer in the neural network")
argument_parser.add_argument("-np", "--n-points", type=int, default=500_000, help="Number of sampled point in the training dataset")
argument_parser.add_argument("-ne", type=int, default=100, help="Number training epochs")
args = argument_parser.parse_args()

np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(args.input_geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

field = IL.fields.Distance(geometry, signed=True, square=False)
sampling_strat = IL.sampling_strategy.CombinedStrategy([
    IL.sampling_strategy.UniformBox(geometry, domain=M.geometry.AABB([-1.2]*geometry.dim, [1.2]*geometry.dim)),
    IL.sampling_strategy.NearGeometryGaussian(geometry)
], [1., 2.])
sampler = IL.PointSampler(geometry, sampling_strat, field)



######## Setup model
model = IL.nn.TailedMultiLayerPerceptron(geometry.dim, args.layer_size, args.n_layers).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")


######## Training
class TMLPTrainer(Trainer):
    def __init__(self,
        lod_weights : torch.Tensor,
        config : TrainingConfig,
    ):
        super().__init__(config)
        self.LoD_weights = lod_weights.reshape((1,-1,1)) 
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        X,Y_target = data
        Y = model.forward_all(X)
        # batch_loss_fit = torch.nn.functional.mse_loss(Y, Y_target.unsqueeze(1).repeat(1, model.n_layers, 1), reduction="none")
        batch_loss_fit = torch.abs(Y - Y_target.unsqueeze(1).repeat(1, model.n_layers, 1))
        return torch.mean(self.LoD_weights * batch_loss_fit)

# pretrainer = SphereInitializeTrainer(10_000, TrainingConfig(
#     BATCH_SIZE=1000,
#     TEST_BATCH_SIZE=5000,
#     N_EPOCHS=1,
#     LEARNING_RATE=1e-4,
#     OPTIMIZER="muon",
#     DEVICE=DEVICE
# ))
# pretrainer.train(model)
# pretrainer.add_callbacks(callbacks.LoggerCB("output/pretraining_log.txt"))

# Setup trainer
lod_weights = torch.ones(model.n_layers, device=DEVICE)
lod_weights[0] = 0.
lod_weights[-1] = 5.

trainer = TMLPTrainer(lod_weights,
    TrainingConfig(
        BATCH_SIZE=10000,
        TEST_BATCH_SIZE = 10000,
        N_EPOCHS=args.ne,
        LEARNING_RATE=1e-3,
        OPTIMIZER="adam",
        DEVICE=DEVICE
))
trainer.set_training_data(TensorDataset(torch.zeros((1,1), device=DEVICE))) # dummy data to be erased by the resampling callback
trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
trainer.add_callbacks(callbacks.ResampleCallback(sampler, args.n_points, device=DEVICE, freq=50, on_ratio=0.4))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 10))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 100, res=200, iso=0.))

trainer.add_scheduler(torch.optim.lr_scheduler.MultiStepLR, milestones=[int(args.ne*0.3), int(args.ne*0.5), int(args.ne*0.7), int(args.ne*0.8), int(args.ne*0.9)], gamma=0.25)
trainer.train(model)


if geometry.dim==2:
    domain = M.geometry.AABB([-1.5,-1.5],[1.5,1.5])
    for i in range(1, model.n_layers):
        model.depth = i
        IL.visualize.render_sdf_2d(None, os.path.join(OUTPUT_DIR, f"contours_depth_{i}.png"), os.path.join(OUTPUT_DIR, f"gradient_depth_{i}.png"),model,domain,DEVICE, batch_size=10_000)

elif geometry.dim==3:
    domain = M.geometry.AABB([-1.2]*3, [1.2]*3)
    res = 400
    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T

    inputs = DataLoader(torch.Tensor(pts).to(DEVICE), batch_size=10_000)
    outputs = []
    grads = []
    inputs = tqdm(inputs, total=len(inputs))
    with torch.inference_mode():
        for batch in inputs:
            v_batch = model.forward_all(batch.to(DEVICE))
            outputs.append(v_batch.detach().cpu())
    dist_values_all = torch.cat(outputs).numpy()
    for i in range(1, model.n_layers):
        dist_values = dist_values_all[:,i].reshape((res,res,res))
        print("Implicit values:", np.amin(dist_values), np.amax(dist_values))

        ### Call marching cubes
        try:
            verts,faces,normals,values = marching_cubes(dist_values, level=0.)
            values = values[:, np.newaxis]
            outmesh = M.mesh.RawMeshData()
            outmesh.vertices += list(verts)
            outmesh.faces += list(faces)
            outmesh = M.mesh.SurfaceMesh(outmesh)
            normal_attr = outmesh.vertices.create_attribute("normals", float, 3, dense=True)
            normal_attr._data = normals
            values_attr = outmesh.vertices.create_attribute("values", float, 1, dense=True)
            values_attr._data = values

            ### Reproject meshes to correct coordinates
            for v in outmesh.id_vertices:
                pV = M.Vec(outmesh.vertices[v])
                ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
                dx, dy, dz = pV.x%1, pV.y%1, pV.z%1

                ixn = ix+1 if ix<res-1 else res-1
                iyn = iy+1 if iy<res-1 else res-1
                izn = iz+1 if iz<res-1 else res-1

                vx = (1-dx)*L[0][ix] + dx * L[0][ixn]
                vy = (1-dy)*L[1][iy] + dy * L[1][iyn]
                vz = (1-dz)*L[2][iz] + dz * L[2][izn]
                outmesh.vertices[v] = M.Vec(vx,vy,vz)
            M.mesh.save(outmesh, os.path.join(OUTPUT_DIR, f"surface_depth_{i}.obj"))

        except ValueError:
            continue
    