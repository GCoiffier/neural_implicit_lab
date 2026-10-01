import os, sys
import mouette as M
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import torch
from torch import nn
from torch.nn import functional as F

import implicitlab as IL
from implicitlab.training import TrainingConfig, Trainer
from implicitlab.training.losses import EikonalLoss
from implicitlab.training import callbacks


argument_parser = argparse.ArgumentParser(
    prog="Medial axis aware learning of signed distance functions",
    description="Fits a neural implicit representation to a neural distance field using an Ambrosio-Tortorelli type approach"
)
argument_parser.add_argument("input_geometry", type=str)
argument_parser.add_argument("-o", "--output-name", default="", type=str)
argument_parser.add_argument("-np", "--n-points", type=int, default=500_000, help="Number of sampled point in the training dataset")
argument_parser.add_argument("-ne", type=int, default=200, help="Number training epochs")
args = argument_parser.parse_args()

np.random.seed(42)

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(args.input_geometry)
print(geometry.geom_type)

DEVICE = IL.utils.get_device()

####### Dataset Sampling

if geometry.dim == 3:
    points, normals = M.sampling.sample_surface(geometry, args.n_points, return_normals=True)
elif geometry.dim == 2:
    points, normals = IL.data.sample_points_and_normals2D(geometry, 100_000)
train_data = IL.data.make_tensor_dataset((points, normals))


class AlphaSigmoid(nn.Module):
    def __init__(self, a=1.):
        super().__init__()
        self.a = a

    def forward(self, x):
        return torch.sigmoid(self.a*x)
    

class PhaseFieldNDF(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        self.mlp = IL.nn.SirenNet(dim_in, 64, 5)
        self.phase_field = nn.Sequential(IL.nn.SirenNet(dim_in, 64, 4), AlphaSigmoid())

    def forward(self, x):
        return self.mlp(x)

model = PhaseFieldNDF(geometry.dim).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")


def AmbrosioTortorelliLoss(inp, out, eps):
    grad = IL.diff_op.gradient(inp, out)
    dir = torch.mean(torch.sum(grad**2, dim=-1))
    norm = torch.mean( (out-1)**2)
    return eps*dir + 1/(4*eps)*norm

class AmbrosioTortorelliTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.EPS = 1e-3
        self.rho = 100.
        self.weights = {
            "on" : 1e3,
            "out" : 100.,
            "normals": 200.,
            "hess": 50.,
            "eikonal" : 50.,
            "AT" : 100.
        }

    def forward_train_batch(self, data, model):
        pts, normals = data
        pts.requires_grad = True
        pts_out = 3*torch.rand_like(pts)-1.5
        pts_out.requires_grad = True

        Y_on = model.mlp(pts)
        batch_loss = self.weights["on"] * torch.mean(torch.abs(Y_on))

        grad_on = torch.autograd.grad(Y_on, pts, grad_outputs=torch.ones_like(Y_on), create_graph=True)[0]
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad_on, normals)

        Y_out = model.mlp(pts_out)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- self.rho * torch.abs(Y_out)))

        PF_out = torch.squeeze(model.phase_field(pts_out))

        grad_out = torch.autograd.grad(Y_out, pts_out, grad_outputs=torch.ones_like(Y_out), create_graph=True)[0]
        batch_loss += self.weights["eikonal"]*EikonalLoss(pts_out, Y_out, grad_out)

        grad_out_sqr = grad_out*grad_out
        hess_times_grad = 0.5 * torch.autograd.grad(
            grad_out_sqr, pts_out,
            grad_outputs=torch.ones_like(grad_out_sqr),
            create_graph=True,   # MUST be True
            retain_graph=True
        )[0]

        loss_hess = torch.pow(PF_out, 2)*torch.sum(hess_times_grad**2, dim=-1)
        # loss_hess = torch.sum(hess_times_grad**2, dim=-1)
        batch_loss += self.weights["hess"]*torch.mean(loss_hess)


        batch_loss += self.weights["AT"]*AmbrosioTortorelliLoss(pts_out, PF_out, self.EPS)
        # loss_hess2 = self.EPS*torch.sum(hess**2,dim=(-2,-1))
        # batch_loss += self.weights["hess"]*torch.mean(loss_hess2)
        
        return batch_loss


class Render2DPhaseFieldCB(IL.training.Callback):

    def __init__(self, 
        save_folder: str, 
        freq: int, 
        plot_domain: M.geometry.AABB = None, 
        resolution: int = 800,  
        output_gradient_norm: bool = True, 
        prefix: str = ""
    ):
        super().__init__()
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        if plot_domain is None:
            self.domain = M.geometry.AABB([-1.5,-1.5],[1.5,1.5])
        else:
            self.domain = plot_domain
        self.freq = freq
        self.res = resolution
        self.output_gradient_norm = output_gradient_norm
        self.prefix = prefix
        if len(self.prefix)>0 and self.prefix[-1]!='_':
            self.prefix += '_'

    def callOnBeginTrain(self, trainer, model):
        contour_path = os.path.join(self.save_folder, self.prefix + f"contour_init.png")
        gradient_path = os.path.join(self.save_folder, self.prefix + f"grad_init.png") if self.output_gradient_norm else None
        phase_field_path = os.path.join(self.save_folder, self.prefix + f"phase_init.png")
        self.render(contour_path, gradient_path, phase_field_path, model, trainer.config.DEVICE, self.res, trainer.config.TEST_BATCH_SIZE)


    def callOnEndEpoch(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        # if self.freq>0 and epoch%self.freq==0:
        if self.freq>0 and epoch%self.freq==0:
            contour_path = os.path.join(self.save_folder, self.prefix + f"contour_{epoch}.png")
            gradient_path = os.path.join(self.save_folder, self.prefix + f"grad_{epoch}.png") if self.output_gradient_norm else None
            phase_field_path = os.path.join(self.save_folder, self.prefix + f"phase_{epoch}.png")
            self.render(contour_path, gradient_path, phase_field_path, model, trainer.config.DEVICE, self.res, trainer.config.TEST_BATCH_SIZE)

    def render(self, contour_path, gradient_path, phase_field_path, model, device, res, batch_size):
        X = np.linspace(self.domain.mini[0], self.domain.maxi[0], res)
        resY = round(res * self.domain.span[1]/self.domain.span[0])
        Y = np.linspace(self.domain.mini[1], self.domain.maxi[1], resY)
        pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T

        if gradient_path is not None:
            dist_values, grad_values = IL.utils.forward_in_batches(
                model.mlp, pts, device, compute_grad=True, batch_size=batch_size)
        phase_values = IL.utils.forward_in_batches(model.phase_field, pts, device, compute_grad=False, batch_size=batch_size)

        ## Contour plot
        img = dist_values.reshape((res,resY)).T
        img = img[::-1,:]
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        n_contours = 16
        plt.contour(img, levels=n_contours, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)
    
        ## Gradient plot
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))
        plt.clf()
        pos = plt.imshow(grad_img, vmin=0., vmax=2., cmap="bwr")
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)
    
        ## Phase plot
        phase_img = phase_values.reshape((res,resY)).T
        phase_img = phase_img[::-1,:]
        plt.clf()
        plt.imshow(phase_img, vmin=0, vmax=1, cmap="Blues_r")
        plt.axis("off")
        plt.savefig(phase_field_path, bbox_inches='tight', pad_inches=0, dpi=200)

# Setup trainer
trainer = AmbrosioTortorelliTrainer(TrainingConfig(
    BATCH_SIZE=1_000,
    TEST_BATCH_SIZE = 50_000,
    N_EPOCHS=args.ne,
    LEARNING_RATE=5e-4,
    OPTIMIZER="adam",
    DEVICE=DEVICE
))


trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
if geometry.dim == 2:
    trainer.add_callbacks(Render2DPhaseFieldCB("output", 50))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 100, res=300, iso=0.))

trainer.set_training_data(train_data)
trainer.train(model)