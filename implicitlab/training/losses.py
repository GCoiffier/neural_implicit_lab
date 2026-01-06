import torch
import torch.nn.functional as F

from ..utils import gradient

class HKRLoss:
    def __init__(self, margin: float = 1e-2, lmbd: float = 10.):
        """
        Hinge Kantorovitch-Rubinstein loss
    
        $$\\text{hKR}(x) = \\max(0, m-x) - \\frac{x}{\\lambda}$$

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

    $$\\mathcal{L}_{\\text{eik}}(x) = (||\\nabla f_\\theta(x)|| -1)^2$$
    
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
    $$\\mathcal{L}_{\\text{heat}}(f) = \\mathbb{E}_x \\left[ e^{-2 \\lambda |f(x)|} \\left( || \\nabla f||^2 + 1 \\right) \\right].$$

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
    """
    pass

    # def singular_hessian_loss(mnfld_points, nonmnfld_points, mnfld_grad, nonmnfld_grad):
    #     nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
    #     nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
    #     mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
    #     mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])

    #     # if dims == 3:
    #     nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
    #     nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

    #     mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
    #     mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)

    #     nonmnfld_det = torch.det(nonmnfld_hessian_term)
    #     mnfld_det = torch.det(mnfld_hessian_term)

    #     morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
    #     morse_nonmnfld = torch.tensor([0.0], device=mnfld_points.device)
    #     # if div_type == 'l1':
    #     morse_nonmnfld = nonmnfld_det.abs().mean()
    #     morse_mnfld = mnfld_det.abs().mean()

    #     morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

    #     return morse_loss    



class ThinPlateLoss:
    """
    
    References:
        - [NeuVAS: Neural Implicit Surfaces for Variational Shape Modeling](https://arxiv.org/abs/2506.13050), Wang et al., 2025
        - [https://github.com/GeometryArt/NeuVAS/tree/main/code](https://github.com/GeometryArt/NeuVAS/tree/main/code)
    """

    def __call__(self, inp_tensor, out_tensor):
        grad = gradient(inp_tensor, out_tensor)               
        gdx  = gradient(inp_tensor, grad[:, 0])
        gdy  = gradient(inp_tensor, grad[:, 1])
        gdz  = gradient(inp_tensor, grad[:, 2])
        hessian = torch.stack((gdx, gdy, gdz), dim=-1)
        mc = self.mean_curvature(hessian, grad)
        gc = self.gaussian_curvature(hessian, grad)
        return torch.abs(4*mc*mc - 2*gc).sum()

    
    def mean_curvature(self, hess, grad):
        grad = grad[:, None, :]
        KM_term_1 = torch.bmm(grad, hess)
        KM_term_1 = torch.bmm(KM_term_1, grad.permute(0, 2, 1)).squeeze(-1)

        hess_diag = torch.diagonal(hess, dim1=-2, dim2=-1)
        trace_hess = torch.sum(hess_diag, dim=-1)[:, None]
    
        grad_norm = grad.norm(dim=-1)
        KM_term_2 = (grad_norm ** 2) * trace_hess

        KM = (KM_term_1 - KM_term_2) / (2 * grad_norm ** 3 + 1e-12)
        KM = torch.abs(KM)
        return KM
    

    def gaussian_curvature(self, hess, grad):
        grad = grad.unsqueeze(0)
        nonmnfld_hessian_term = torch.cat((hess.unsqueeze(0), grad[:, :, :, None]), dim=-1)
        zero_grad = torch.zeros(
            (grad.shape[0], grad.shape[1], 1, 1),
            device=grad.device)
        zero_grad = torch.cat((grad[:, :, None, :], zero_grad), dim=-1)
        nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
        Kg = (-1. / (grad.norm(dim=-1) ** 4)) * torch.det(nonmnfld_hessian_term)
        Kg = torch.abs(Kg.permute(1, 0))
        return Kg