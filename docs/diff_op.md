---
title: Differential Operators
---

Based on [https://github.com/vsitzmann/siren/blob/master/diff_operators.py](https://github.com/vsitzmann/siren/blob/master/diff_operators.py)

!!! note
The convention is that the input tensor always comes before the output tensor. For instance, computing the gradient of Y with relation to X is written `gradient(X,Y)` and **not** `gradient(Y,X)`.

:::implicitlab.utils
    options:
        heading_level: 2