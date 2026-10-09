---
title: Neural Network Architectures
---

This page lists the neural network models and components that are available in the library.

## Regular Architectures

:::implicitlab.nn.mlp
    options:
        heading_level: 3

:::implicitlab.nn.siren
    options:
        heading_level: 3
        members:
            - SirenNet

<!-- :::implicitlab.nn.phase
    options:
        heading_level: 3 -->

:::implicitlab.nn.quanet
    options:
        heading_level: 3
    members:
        - QuaNet

## Lipschitz Architectures

Lipschitz neural networks are neural networks where the Lipschitz constant is controlled (equal or inferior to a given value). Recall that a function $f$ is $K$-Lipschitz if for any points $x,y \in \mathbb{R}^n$, we have:

$$|f(x) - f(y)| \leqslant K\,||x -y||.$$

The signed distance function is notoriously a 1-Lipschitz function, and using 1-Lipschitz neural implicit enables SDF approximations with guarantees as well as the use of different loss functions (see for instance the [`hKRTrainer`][implicitlab.training.trainers.hkr.hKRTrainer] class).

:::implicitlab.nn.lipschitz
    options:
        heading_level: 3


## Positional encodings

Positional encodings are fixed transformations applied to the input data before being fed into a neural network. They increase the model's accuracy by tackling low-frequency biases. When defining a neural model, simply use a `torch.nn.Sequential` object to add your encoding before your neural network. Note that the encoding size and the input dimension of the network should match:

```python
# Add an encoding made of Fourier Features to a simple multi-layer perceptron
model = torch.nn.Sequential(
    RandomFourierEncoding(geometry,1000),
    MultiLayerPerceptron(1000, 128, 4)
)
```

:::implicitlab.nn.encodings
    options:
        heading_level: 3



## Utilities

:::implicitlab.nn.utils
    options:
        heading_level: 3
