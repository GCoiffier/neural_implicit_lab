import os, sys
import mouette as M
import torch
import numpy as np

import implicitlab as IL
from implicitlab.training.losses import EikonalLoss, ThinPlateLoss
from implicitlab.training import TrainingConfig,Trainer
from implicitlab.training import callbacks

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

points = np.asarray(geometry.vertices)
pc = M.mesh.from_arrays(points)
M.mesh.save(pc, "output/train_pts.geogram_ascii")
train_data = IL.data.make_tensor_dataset((points,), DEVICE)

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
        pts, = data
        pts.requires_grad = True
        Y_on = model(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        pts_out = 3*torch.rand_like(pts)-1
        pts_out.requires_grad = True
        Y_out = model(pts_out)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(Y_out)))

        # batch_loss += self.weights["thinplate"] * ThinPlateLoss()(pts, Y_on)
        
        batch_loss += self.weights["eikonal"] * EikonalLoss()(pts_out, Y_out)        
        return batch_loss


trainer = ThinPlateTrainer(TrainingConfig(
    BATCH_SIZE=200,
    TEST_BATCH_SIZE = 5000,
    N_EPOCHS=1200,
    LEARNING_RATE=1e-4,
    DEVICE=DEVICE
))

trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 50))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 300, res=100, iso=0.))
trainer.set_training_data(train_data)
trainer.train(model)

IL.nn.save_model(model, "output/model.pt")