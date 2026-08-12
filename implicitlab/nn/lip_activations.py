import numpy as np
import torch
import torch.nn as nn

"""
Activations taken from:  
- https://github.com/deel-ai/orthogonium/blob/main/orthogonium/layers/custom_activations.py  
- https://github.com/singlasahil14/SOC/blob/main/custom_activations.py  
"""

SQRT_2 = np.sqrt(2)

class Abs(nn.Module):
    """Absolute value activation function. Defined as `torch.abs(z)` for an input vector `z`."""
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return torch.abs(z)


class SoftHuber(nn.Module):
    def __init__(self, delta=0.05):
        """
        SoftHuber activation function, i.e. differentiable approximation of the Huber loss. 
        
        This function behaves like abs(x) far from zero and like x**2 near zero. The transition between these two
        behaviors is controlled by the delta parameter.

        Args:
            delta (float, optional): The threshold at which to switch between L1 and L2 loss. Defaults to 0.05.
        """
        super(SoftHuber, self).__init__()
        self.delta = delta

    def forward(self, z):
        # we dont multiply by delta**2 in order to have a Lipschitz constant of 1
        return self.delta * (torch.sqrt(1 + (z / self.delta) ** 2) - 1)


class Householder(nn.Module):
    def __init__(self, channels, axis=-1):
        """
        A activation that applies a parameterized transformation via Householder
        reflection technique. It is initialized with the number of input channels, which must
        be even, and an axis that determines the dimension along which operations are applied.
        This is a corrected version of the original implementation from Singla et al. (2019),
        which features a 1/sqrt(2) scaling factor to be 1-Lipschitz.

        Attributes:
            theta (torch.nn.Parameter): Learnable parameter that determines the transformation
                applied via Householder reflection.
            axis (int): Dimension along which the operation is performed.

        Args:
            channels (int): Total number of input channels. Must be an even number.
            axis (int): Dimension along which the transformation is applied. Default is last channel (-1).
        """
        super(Householder, self).__init__()
        assert (channels % 2) == 0
        eff_channels = channels // 2

        self.theta = nn.Parameter(
            0.5 * np.pi * torch.ones(1, eff_channels), requires_grad=True
        )
        self.axis = axis

    def forward(self, z):
        theta = self.theta
        x, y = z.split(z.shape[self.axis] // 2, self.axis)

        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))

        a_2 = x * torch.cos(theta) + y * torch.sin(theta)
        b_2 = x * torch.sin(theta) - y * torch.cos(theta)

        a = x * (selector <= 0) + a_2 * (selector > 0)
        b = y * (selector <= 0) + b_2 * (selector > 0)
        return torch.cat([a, b], dim=self.axis) / SQRT_2
