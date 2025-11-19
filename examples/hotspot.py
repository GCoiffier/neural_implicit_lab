import os, sys
import mouette as M
import torch

import implicitlab as IL
from implicitlab.training.losses import EikonalLoss
from implicitlab.training import TrainingConfig,Trainer
from implicitlab.training import callbacks

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

if geometry.dim == 3:
    points, normals = M.sampling.sample_surface(geometry, 300_000, return_normals=True)
elif geometry.dim == 2:
    points, normals = IL.data.sample_points_and_normals2D(geometry, 100_000)
train_data = IL.data.make_tensor_dataset([points, normals], DEVICE)

M.mesh.save(M.procedural.vector_field(points, normals, length_mult=0.1), "output/train_pts.geogram_ascii")


###### Training 

# Setup model
model = IL.nn.MultiLayerPerceptron(geometry.dim, 128, 5).to(DEVICE)
# model = IL.nn.SirenNet(geometry.dim, 128, 5).to(DEVICE)
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
            "on" : 10.,
            "eikonal": 1.,
            "normals": 0.1,
            "heat": 1.
        }

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts,normals = data
        pts.requires_grad = True
        Y_on = model(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        pts_out = 2.1*torch.rand_like(pts)-1
        pts_out.requires_grad = True
        Y_out = model(pts_out)

        batch_grad = torch.autograd.grad(Y_out, pts_out, grad_outputs=torch.ones_like(Y_out), create_graph=True)[0]
        # batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(batch_grad, normals)

        batch_grad_norm = batch_grad.norm(dim=-1)
        batch_loss += self.weights["heat"] * torch.mean(torch.exp(-2*self.lmbd*torch.abs(Y_out))*(1 + batch_grad_norm**2))
        batch_loss += self.weights["eikonal"] * torch.nn.functional.mse_loss(batch_grad_norm, torch.ones_like(batch_grad_norm))
        
        return batch_loss

config = TrainingConfig(
    BATCH_SIZE=100,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE
)


trainer = HotspotTrainer(config, lmbd=1.)

trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 10))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 10, res=300, iso=0.))
trainer.set_training_data(train_data)
trainer.train(model)

IL.nn.save_model(model, "output/model.pt")