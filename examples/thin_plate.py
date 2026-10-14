import os, sys
import mouette as M
import torch
import numpy as np

import implicitlab as IL
from implicitlab.training.losses import EikonalLoss, ThinPlateLoss
from implicitlab.training import TrainingConfig,Trainer
from implicitlab.training import callbacks

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

points = np.asarray(geometry.vertices)
pc = M.mesh.from_arrays(points)
M.mesh.save(pc, os.path.join(OUTPUT_DIR, "train_pts.geogram_ascii"))
train_data = IL.data.make_tensor_dataset((points,))

###### Training 

# Setup model
model = IL.nn.MultiLayerPerceptron(geometry.dim, 128, 6, activ=torch.nn.Softplus).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")

# Setup trainer
class ThinPlateTrainer(Trainer):

    def __init__(self, 
        config : TrainingConfig
    ):
        super().__init__(config)
        self.rho = 100.
        self.weights = {
            "eikonal" : 50.,
            "on" : 7000.,
            "out" : 600.,
            "thinplate": 1.,
        }
    
    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts_on, = data
        pts_on.requires_grad = True
        pts_out = 3*torch.rand_like(pts_on)-1.5
        pts_out.requires_grad = True
        pts_all = torch.cat((pts_on, pts_out))
        n = pts_on.shape[0]

        values = model(pts_all)
        grad = torch.autograd.grad(values.sum(), pts_all, create_graph=True)[0]
        
        values_on, values_out = values[:n], values[n:]
        grad_on, grad_out = grad[:n], grad[n,:]
        batch_loss = self.weights["on"] * torch.mean(torch.abs(values_on))
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(values_out)))
        batch_loss += self.weights["thinplate"] * ThinPlateLoss()(pts_on, values_on, grad_on)
        batch_loss += self.weights["eikonal"] * EikonalLoss(pts_out, values_out, grad_out)  
        return batch_loss


trainer = ThinPlateTrainer(TrainingConfig(
    BATCH_SIZE = 2**12,
    TEST_BATCH_SIZE = 2**15,
    N_EPOCHS = 5000,
    LEARNING_RATE = 1e-4,
    DEVICE = DEVICE
))

trainer.add_callbacks(callbacks.PrinterCB())
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 500))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 500, res=256, iso=0.))
trainer.set_training_data(train_data)
trainer.train(model)

model.save(os.path.join(OUTPUT_DIR, "model.inr"))