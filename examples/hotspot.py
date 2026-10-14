import os, sys
import mouette as M
import torch
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer
from implicitlab.training import callbacks

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

points, normals = IL.data.sample_geometry_with_normals(geometry, 2**18)
train_data = IL.data.make_tensor_dataset([points, normals])

# M.mesh.save(M.procedural.vector_field(points, normals, length_mult=0.1), "output/train_pts.geogram_ascii")

###### Training 

# Setup model
model = IL.nn.SirenNet(geometry.dim, 128, 5).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
class HotspotTrainer(Trainer):

    def __init__(self, 
        config : TrainingConfig,
        lmbd = 10.
    ):
        super().__init__(config)
        self.lmbd = lmbd
        self.weights = {
            "on" : 50.,
            "eikonal": 0.1,
            "normals": 1.,
            "heat": 1.
        }
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts_on,normals = data
        n = pts_on.shape[0]
        pts_out = 3.*torch.rand_like(pts_on)-1.5

        pts_all = torch.cat((pts_on, pts_out))
        pts_all.requires_grad = True
        values = model(pts_all)
        grad = torch.autograd.grad(values.sum(), pts_all, create_graph=True)[0]
        
        values_on, values_out = values[:n], values[n:]
        grad_on, grad_out = grad[:n], grad[n:]

        batch_loss = self.weights["on"] * torch.mean(torch.abs(values_on))
        batch_loss += self.weights["normals"]*F.mse_loss(grad_on, normals)
        batch_grad_norm = grad_out.norm(dim=-1)
        batch_loss += self.weights["heat"] * torch.mean(torch.exp(-2*self.lmbd*torch.abs(values_out))*(1 + batch_grad_norm**2))
        batch_loss += self.weights["eikonal"] * F.mse_loss(batch_grad_norm, torch.ones_like(batch_grad_norm))
        
        return batch_loss

config = TrainingConfig(
    BATCH_SIZE = 2**14,
    TEST_BATCH_SIZE = 2**17,
    N_EPOCHS=1000,
    OPTIMIZER="adam",
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE
)


trainer = HotspotTrainer(config, lmbd=1.)

trainer.add_callbacks(callbacks.PrinterCB())
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB(OUTPUT_DIR, 100))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB(OUTPUT_DIR, 100, res=256, iso=0.))
trainer.set_training_data(train_data)
trainer.train(model)

model.save(os.path.join(OUTPUT_DIR, "model.inr"))