import torch
import torch.nn.functional as F

class HKRLoss:
    def __init__(self, margin: float = 1e-2, lmbd: float = 10.):
        """
        Hinge Kantorovitch-Rubinstein loss
    
        $$\\text{hKR}(x) = \lambda*\max(0, m-x) - \\frac{x}{\lambda}$$

        Args:
            margin (float, optional):  hinge margin. Must be small but not too small. Defaults to 1e-2.
            lmbd (float, optional): weight balance between the two terms.. Defaults to 10.

        References:
            - _Achieving robustness in classification using optimal transport with hinge regularization_, Serrurier et al., 2021
            - _1-Lipschitz neural distance fields_, Coiffier and Béthune, 2024
        """
        self.margin = margin  # must be small but not too small.
        self.lmbd   = lmbd  # must be high.

    def __call__(self, y):
        """
        Args:
            y (torch.Tensor): vector of predicted values
        """
        return  F.relu(self.margin - y) + (1./self.lmbd) * torch.mean(-y)
    
class VectorAlignmentLoss:
    """Cosine similarity loss between two vectors."""
    def __init__(self): pass

    def __call__(self, y, target):
        """
        Args:
            y (torch.Tensor): input tensor
            target (torch.Tensor): target tensor

        Returns:
            torch.Tensor: Cosine similarity
        """
        return (1-F.cosine_similarity(y, target, dim = 1)*2).mean()


class EikonalLoss:
    """
    The Eikonal loss regularizes the gradient of a neural implicit to have unit norm everywhere:

    $$(||\\nabla f_\theta|| -1)^2$$
    
    The gradient is computed from the input $X$ and the output $Y$ of the neural model.
    """
    def __init__(self): pass

    def __call__(self, X, Y):
        """
        Args:
            X (torch.Tensor): input batch of the model
            Y (torch.Tensor): output of the model.

        Returns:
            torch.Tensor: Eikonal loss
        """
        batch_grad = torch.autograd.grad(Y, X, grad_outputs=torch.ones_like(Y), create_graph=True)[0]
        batch_grad_norm = batch_grad.norm(dim=-1)
        return F.mse_loss(batch_grad_norm, torch.ones_like(batch_grad_norm))


class SALLoss:
    def __init__(self, l=1., metric="l2"):
        """
        Signed agnostic learning loss.

        Args:
            l (float, optional): Power value for the distance. Defaults to 1.
            metric (str, optional): which metric to use. Choices are "l2" and "l0". Defaults to "l2".
        
        References:
            _SAL: Sign Agnostic Learning of Shapes From Raw Data_, Atzmon and Lipman, 2020 
        """
        self.l = l

        self.callfun = {
            "l2" : self.SAL_l2,
            "l0" : self.SAL_l0
        }.get(metric, self.SAL_l2)

    def __call__(self, y_pred, y_target):
        """
        Args:
            y_pred (torch.Tensor): predicted values by the model
            y_target (torch.Tensor): target values
        """
        return self.callfun(y_pred, y_target)

    def SAL_l2(self,y_pred,y_target):
        return torch.mean(torch.abs(torch.abs(y_pred) - y_target)**self.l)
    
    def SAL_l0(self,y_pred,y_target):
        return torch.mean(torch.abs(torch.abs(y_pred) - 1)**self.l)
    

class SALDLoss:
    """
    Sign agnostic learning loss with derivatives

    References:
        _SALD: Sign Agnostic Learning with Derivatives_, Atzmon and Lipman, 2020
    """
    def __init__(self): pass

    def __call__(self,y_pred,y_target):
        """
        Args:
            y_pred (torch.Tensor): predicted values by the model
            y_target (torch.Tensor): target values
        """
        return torch.min( torch.norm(y_pred-y_target), torch.norm(y_pred+y_target))
    


class HotspotLoss:
    """
    $$e^{-2 \lambda |f(x)| \left( || \\nabla f||^2 + 1 \\right)}$$

    References:
        - HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition, Wang et al., 2025
    """
    def __init__(self, lmbd:float ):
        self.lmbd = lmbd  

    def __call__(self, X, Y):
        batch_grad = torch.autograd.grad(Y, X, grad_outputs=torch.ones_like(Y), create_graph=True)[0]
        batch_grad_norm = batch_grad.norm(dim=-1)
        return torch.mean(torch.exp(-2*self.lmbd*torch.abs(Y))*(1 + batch_grad_norm**2))
    

class SingularHessianLoss:
    """
    TODO

    def singular_hessian_loss(mnfld_points, nonmnfld_points, mnfld_grad, nonmnfld_grad):
        nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
        nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
        mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
        mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])

        # if dims == 3:
        nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
        nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

        mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
        mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)

        nonmnfld_det = torch.det(nonmnfld_hessian_term)
        mnfld_det = torch.det(mnfld_hessian_term)

        morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
        morse_nonmnfld = torch.tensor([0.0], device=mnfld_points.device)
        # if div_type == 'l1':
        morse_nonmnfld = nonmnfld_det.abs().mean()
        morse_mnfld = mnfld_det.abs().mean()

        morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

        return morse_loss
    """

    