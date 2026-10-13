import numpy as np
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.measure import marching_cubes
from ..utils import forward_in_batches
from .queries import estimate_local_Lipschitz_constant


def generate_regular_grid(domain : M.geometry.AABB, shape:tuple, device: torch.device) -> torch.Tensor:
    L = [torch.linspace(domain.mini[i], domain.maxi[i], n, dtype=torch.float32, device=device) for i,n in enumerate(shape)]
    pts = torch.stack(torch.meshgrid(*L, indexing="ij"), dim=-1).reshape(-1, len(shape))
    return pts

def prepare_model(model, device):
    try:
        _model = torch.compile(model.eval().to(device))
    except:
        _model = model.eval().to(device)
    return _model

def render_sdf_2d(render_path, contour_path, gradient_path, model, domain: M.geometry.AABB, device : torch.device, res=1000, batch_size: int=2**15, **kwargs):
    """ Renders a 2D SDF

    Args:
        render_path (str): Path to export a color render
        contour_path (str): Path to export the contour plot
        gradient_path (str): Path to export the gradient norm plot
        model : SDF neural model
        domain (M.geometry.AABB): axis aligned bounding box of dimension 2to represent the plotting domain.
        device (torch.device): device onto which the computation are performed.
        res (int, optional): Image resolution. Defaults to 800.
        batch_size (int, optional): Size of forward batches. Defaults to 2**15.

    Keyword Args:
        n_contours (int, optional): Defaults to 16.    
    """
    assert domain.dim == 2
    _model = prepare_model(model, device)
    resY = round(res * domain.span[1]/domain.span[0])
    pts = generate_regular_grid(domain, (res, resY), device)
    if gradient_path is not None:
        dist_values,grad_values = forward_in_batches(
            _model, pts, device, 
            compute_grad=True, batch_size=batch_size)
        grad_values = grad_values.cpu().numpy()
    else:
        dist_values = forward_in_batches(_model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.cpu().numpy()
    img = dist_values.reshape((res,resY)).T
    img = img[::-1,:]

    vmin = min(np.amin(img), -1e-8)
    vmax = max(np.amax(img), 1e-8)

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm)
        plt.axis('off')
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        n_contours = kwargs.get("n_contours", 16)
        plt.contour(img, levels=n_contours, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if gradient_path is not None:
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

def render_sdf_quad(render_path, contour_path, gradient_path, model, P0, P1, P2, device: torch.device, res: int =1000, batch_size: int = 2**15, **kwargs):
    """
    Renders a 3D sdf along a specified (planar) quad in space.

    Args:
        render_path (str): Path to export a color render
        contour_path (str): Path to export the contour plot
        gradient_path (str): Path to export the gradient norm plot
        model : SDF neural model
        P0 (torch.Tensor): First point of the quad ( (0,0) in parameter space)
        P1 (torch.Tensor): Second point of the quad ( (1,0) in parameter space)
        P2 (torch.Tensor): Third point of the quad ( (0,1) in parameter space)
        device (torch.device): device onto which the computation are performed.
        res (int, optional): Image resolution. Defaults to 1000.
        batch_size (int, optional): Size of forward batches. Defaults to 2**15.
    """
    _model = prepare_model(model, device)
    P0 = torch.as_tensor(P0, device=device)
    dx = torch.as_tensor(P1 - P0, device=device)
    dy = torch.as_tensor(P2 - P0, device=device)
    resY = round(res * M.geometry.norm(dy)/M.geometry.norm(dx))
    pts = generate_regular_grid(M.geometry.AABB((0.,0.), (1.,1.)), (res,resY), device) # flat unit grid
    pts = P0 + pts[:,0:1]*dx + pts[:,1:2]*dy # apply linear transform to map the unit grid to the quad
    if gradient_path is not None:
        dist_values,grad_values = forward_in_batches(
            _model, pts, device, 
            compute_grad=True, batch_size=batch_size)
        grad_values = grad_values.cpu().numpy()
    else:
        dist_values = forward_in_batches(_model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.cpu().numpy()

    img = dist_values.reshape((res,resY)).T
    img = img[::-1,:]

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1
    print(vmin, vmax)

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm)
        plt.axis('off')
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=kwargs.get("n_levels", 16), colors='k', linestyles="solid", linewidths=0.5)
        plt.contour(img, levels=[kwargs.get("zero_contour_offset", 0.)], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=kwargs.get("dpi", 500))

    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr")
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)


def render_lip_constant_2d(img_path: str, model, domain : M.geometry.AABB, device : torch.device, res=1000, batch_size=2*15):
    """Renders a 2D image of the local Lipschitz constant (maximal gradient norm in a neighborhood) of a 2D model.

    Args:
        img_path (str): path to export the render.
        model (torch.nn.Module): the neural model
        domain (M.geometry.AABB): domain onto which the function values are computed.
        device (torch.device): device onto which the computation are performed.
        res (int, optional): Image resolution. Defaults to 1000.
        batch_size (int, optional): batch size for forward computation. Defaults to 2**15.
    """
    assert domain.dim == 2
    _model = prepare_model(model, device)
    resY = round(res * domain.span[1]/domain.span[0])
    pts = generate_regular_grid(domain, (res, resY), device)
    lip_values = estimate_local_Lipschitz_constant(_model, pts, device, batch_size=batch_size)
    img = lip_values.reshape((res,resY)).T
    img = img[::-1,:]
    plt.clf()
    pos = plt.imshow(img, cmap="Blues_r")
    plt.axis('off')
    plt.colorbar(pos)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)


def reconstruct_surface_marching_cubes(
    model : torch.nn.Module, 
    domain : M.geometry.AABB, 
    device : torch.device, 
    iso : int = 0, 
    res : int = 256, 
    batch_size : int = 2**16,
    **kwargs
) -> dict:
    """Extracts isosurfaces of a given neural implicit by using the Marching Cube algorithm.

    Args:
        model (torch.nn.Module): the neural implicit function.
        domain (M.geometry.AABB): domain onto which the function values are computed.
        device (torch.device): device onto which the computation are performed.
        iso (int, optional): value of the isosurface. Several values can be provided in a list. If that is the case, one mesh per isovalue will be produced. Defaults to 0.
        res (int, optional): resolution of the marching cubes grid. Defaults to 100.
        batch_size (int, optional): batch size for forward computation. Defaults to 2**16.

    Keyword Args:
        use_tqdm (bool, optional): whether to display a progress bar during computation. Defaults to False.
        full_precision (bool, optional): by default, the computation is done over torch.float16 for faster inference. This flag enables computation in full torch.float32 precision. Defaults to False.

    Returns:
        dict: returns a dictionary of ((i,iso), mesh) where i is a unique identifier, iso is the value of the isovalue and mesh is the output surface mesh (mouette.mesh.SurfaceMesh type).

    References:
        - _Marching cubes: A high resolution 3D surface construction algorithm_, Lorensen & Cline (1998)  
        - https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_marching_cubes.html  
    """
    if isinstance(iso, (int,float)): iso = [iso]
    _model = prepare_model(model, device)
    
    ### Feed grid to model
    pts = generate_regular_grid(domain, (res, res, res), device)
    dist_values = forward_in_batches(
        _model, pts, device,
        compute_grad=False, batch_size=batch_size, 
        use_tqdm=kwargs.get("use_tqdm", False),
        full_precision=kwargs.get("full_precision", False)
    )
    dist_values = dist_values.reshape((res,res,res)).cpu().numpy()
    print("Implicit values:", np.amin(dist_values), np.amax(dist_values))

    ### Call marching cubes
    to_save = dict()
    spacing = tuple((domain.maxi - domain.mini) / (res - 1))
    for ioff,off in enumerate(iso):
        try:
            verts,faces,normals,values = marching_cubes(dist_values, level=off, spacing=spacing)
        except ValueError:
            continue
        verts += domain.mini
        m = M.mesh.from_arrays(verts, F=faces)
        m.vertices.register_array_as_attribute("normals",normals)
        m.vertices.register_array_as_attribute("values", values)
        to_save[(ioff, off)] = m
    return to_save