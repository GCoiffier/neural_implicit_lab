---
title: Full example
---


This tutorial goes into a full length example on how to train a SIREN network on a given input mesh to get a neural implicit surface. See [the example folder](https://github.com/GCoiffier/neural_implicit_lab/tree/main/examples) for other full examples.

The plan is to import some geometrical object $M$ that lives in a domain $D \subset \mathbb{R}^n$ and optimize for four loss functions on it:

$$\mathcal{L}_{\text{Dirichlet}}[f]  = \int_M |f_\theta(x)| dx$$

$$\mathcal{L}_{\text{normal}}[f] = \int_M  ||\nabla f(x) - n(x)||\;dx$$

$$\mathcal{L}_{\text{Eikonal}}[f] = \int_D (|| \nabla f(x)|| - 1)^2\;dx$$

$$\mathcal{L}_{\text{out}}[f] = \int_D \exp(-\alpha|f(x)|)\;dx$$


## Package imports
We begin by importing everything that we need and loading the input geometry
```python
import os, sys
import mouette as M
import torch
from torch import nn
import numpy as np

import implicitlab as IL
from implicitlab.training import TrainingConfig, ImplicitSurfaceTrainer
from implicitlab.training import callbacks

np.random.seed(42)

os.makedirs("output", exist_ok=True)
geometry = IL.load_geometry(sys.argv[1])
geometry = IL.data.extract_boundary_polyline_from_2D_mesh(geometry) # make sure that if the geometry is 2D, then it is a polyline
print(geometry.geom_type)
```

We then define the device on which the computation will be made.
```python
DEVICE = IL.utils.get_device()
```
This utility function returns `"cuda"` if a compatible GPU was detected on the machine, otherwise runs on the CPU. Multi-GPU optimization are not supported natively.

## Sampling the dataset
The dataset for this example is made of two components:  
- points on the geometry and their normals (for $\mathcal{L}_{\text{Dirichlet}}$ and $\mathcal{L}_{\text{normal}}$)  
- points outside of the geometry (for $\mathcal{L}_{\text{out}}$ and $\mathcal{L}_{\text{Eikonal}}$)

We rely on the simple geometry sampling utility function for the first part, and on a point sampler object for the second. We will consider 66% of the points in a bounding box $[-1.1, 1.1]^d$, and the remaining 33% as points near the surface. This distribution helps to converge faster in practice.

```python
points_on, normals = IL.data.sample_geometry_with_normals(geometry, 100_000)

sampler = IL.data.PointSampler(
    geometry,
    IL.sampling_strategies.CombinedStrategy([
        IL.sampling_strategies.UniformBox(geometry, domain=M.geometry.AABB([-1.1, -1.1], [1.1, 1.1])),
        IL.sampling_strategies.NearGeometryGaussian(geometry, stdv=0.02)
    ], [2., 1.]),
    # no field attached
)
points_out = sampler.sample(100_000, on_ratio=0.) # no need to sample on the geometry here
train_data = IL.data.make_tensor_dataset((points_out, points_on, normals)) # Compiles everything into a TensorDataset object
```

## Setup the model

We then move on to defining a pytorch model to be trained. We use a SIREN network for this problem, but any `torch.nn.Module` object can work here.

```python
model = IL.nn.SirenNet(geometry.dim, 128, 5).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")
```

## Setup the trainer

We then need to define the trainer. The idea is to inherit from the base `Trainer` class, which defines the training loop, and overwrite the `forward_training_batch` method:


```python
class ExampleTrainer(IL.training.Trainer):

    def __init__(self, 
        config : TrainingConfig
    ):
        super().__init__(config)
        self.weights = {
            "dirichlet" : 700.,
            "eikonal" : 5.,
            "out" : 60.,
            "normals": 10.,
        } # follow the weights prescribed in the SIREN paper
    
    def forward_test_batch(self, data, model): pass
    
    def forward_train_batch(self, data, model):
        pts_out, pts_on, normals = data
        pts_out.requires_grad = True # to compute their gradient for the eikonal loss
        pts_on.requires_grad = True # to compute their gradient to be aligned with the normals

        value_on = model(pts_on)
        batch_loss = self.weights["dirichlet"] * torch.mean(torch.abs(value_on))

        grad_on = IL.diff_op.gradient(pts_on, value_on)
        batch_loss += self.weights["normals"]*torch.nn.functional.mse_loss(grad_on, normals)

        value_out = model(pts_out)
        batch_loss += self.weights["out"] * torch.mean(torch.exp(- 100. torch.abs(value_out)))
        batch_loss += self.weights["eikonal"] * IL.training.losses.EikonalLoss(pts_out, value_out)        
        return batch_loss
```

The trainer can then be instantiated by specifying a training configuration:

```python
trainer = ExampleTrainer(TrainingConfig(
    BATCH_SIZE=10_000,
    TEST_BATCH_SIZE = 50_000,
    N_EPOCHS=200,
    LEARNING_RATE=1e-3,
    OPTIMIZER="Adam",
    DEVICE=DEVICE,
    NUM_DATALOADER_WORKERS=-1 # set to the max number of torch threads
))
```

## Adding callbacks

To visualize the result of the training, we can add callbacks to our trainer:

```python
# the logger will write the loss after each epoch on a text file
trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt")) 
if geometry.dim == 2:
    # output 2D images of the implicit field
    trainer.add_callbacks(callbacks.Render2DCB("output", 50)) 
elif geometry.dim == 3:
    # output a mesh reconstruction of the 0 level set 
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 50, res=400, iso=0.)) 
```

## Run the optimization

The only thing remaining is to give the training dataset to the trainer and start the optimization:

```python
trainer.set_training_data(train_data)
trainer.train(model)
```