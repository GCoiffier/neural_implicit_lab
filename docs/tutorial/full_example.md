---
title: Full example
---


This tutorial goes into a full length example on how to train a SIREN network on a given input mesh to get a neural implicit surface. See [this example file](https://github.com/GCoiffier/neural_implicit_lab/blob/main/examples/implicit_surface.py) for the full code.

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
print(geometry.geom_type)
```

We then define the device on which the computation will be made.
```python
DEVICE = IL.utils.get_device()
```
This utility function returns `"cuda"` if a compatible GPU was detected on the machine, otherwise runs on the CPU. Multi-GPU optimization are not supported natively.

## Sampling the dataset
As such, we sample points and normals onto our geometry for an attach term. Points for the eikonal and the outside loss will be sampled randomly at each training (see below).


```python
if geometry.dim == 3:
    points, normals = M.sampling.sample_surface(geometry, 100_000, return_normals=True) # invoke mouette's sampling algorithm
elif geometry.dim == 2:
    points, normals = IL.data.sample_points_and_normals2D(geometry, 100_000) # special utility for 2D objects (not available in mouette)
train_data = IL.data.make_tensor_dataset((points, normals)) # Compiles everything into a TensorDataset object
```

## Setup the model

```python
model = IL.nn.SirenNet(geometry.dim, 128, 5).to(DEVICE)
print(f"{IL.nn.count_parameters(model)} parameters")
```

## Setup the trainer

```python
trainer = ImplicitSurfaceTrainer(TrainingConfig(
    BATCH_SIZE=10_000,
    TEST_BATCH_SIZE = 50_000,
    N_EPOCHS=args.ne,
    LEARNING_RATE=1e-3,
    OPTIMIZER="Adam",
    DEVICE=DEVICE
))

trainer.add_callbacks(callbacks.LoggerCB("output/training_log.txt"))
if geometry.dim == 2:
    trainer.add_callbacks(callbacks.Render2DCB("output", 50))
elif geometry.dim == 3:
    trainer.add_callbacks(callbacks.MarchingCubeCB("output", 50, res=400, iso=0.))
trainer.set_training_data(train_data)
```

```python
trainer.train(model)
```