---
title: Neural Network Architectures
---

This page lists the neural network models and components that are available in the library.

## Regular Architectures

:::implicitlab.nn.mlp
    options:
        heading_level: 3
        inherited_members: false

:::implicitlab.nn.siren
    options:
        heading_level: 3
        inherited_members: false
        members:
            - SirenNet

<!-- :::implicitlab.nn.phase
    options:
        heading_level: 3 -->

:::implicitlab.nn.tailed_mlp
    options:
        heading_level: 3
        inherited_members: false

:::implicitlab.nn.quanet
    options:
        heading_level: 3
        inherited_members: false
        members:
            - QuaNet

## Lipschitz Architectures

Lipschitz neural networks are neural networks where the Lipschitz constant is controlled (equal or inferior to a given value). Recall that a function $f$ is $K$-Lipschitz if for any points $x,y \in \mathbb{R}^n$, we have:

$$|f(x) - f(y)| \leqslant K\,||x -y||.$$

The signed distance function is notoriously a 1-Lipschitz function, and using 1-Lipschitz neural implicit enables SDF approximations with guarantees as well as the use of different loss functions (see for instance the [`hKRTrainer`][implicitlab.training.trainers.hkr.hKRTrainer] class).

:::implicitlab.nn.lipschitz
    options:
        heading_level: 3
        inherited_members: false
        filters:
            - "!parameter_singular_values"


## Utilities

:::implicitlab.nn.utils
    options:
        heading_level: 3
        inherited_members: false


:::implicitlab.nn.lipschitz
    options:
        heading_level: 3
        members:
            - "parameter_singular_values"