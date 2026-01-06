import numpy as np
import mouette as M
import torch
from torch.utils.data import DataLoader
from scipy.spatial import KDTree
from tqdm import trange
from ..utils import forward_in_batches

def project_onto_iso(
    query_pts : np.ndarray, 
    model : torch.nn.Module, 
    iso : float = 0., 
    device : str = "cpu", 
    batch_size : int = 1000, 
    normalize_grad : bool = False,
    max_steps : int = 100, 
    precision : float = 1e-6
) -> np.ndarray:
    """Project the input points onto the nearest point of a given isosurface.

    Args:
        query_pts (np.ndarray): input point
        model (torch.nn.Module): the neural implicit function.
        iso (float, optional): which isovalue the points should be projected to. Defaults to 0..
        device (str, optional): device onto which the computation is performed. Defaults to "cpu".
        batch_size (int, optional): batch size for forward/backward computation on the neural implicit. Defaults to 1000.
        normalize_grad (bool, optional): whether to normalize the neural implicit gradient at each steps. Defaults to False.
        max_steps (int, optional): maximum number of steps. Defaults to 100.
        precision (float, optional): . Defaults to 1e-6.

    Returns:
        np.ndarray: the projected points
    """
    query_tnsr = torch.Tensor(query_pts).to(device)
    model = model.to(device)
    loader = DataLoader(query_tnsr, batch_size=batch_size)
    output = []
    for batch in loader:
        for _ in range(max_steps):
            batch.requires_grad = True
            dist = model(batch) - iso
            dist.backward(torch.ones_like(dist))
            grad = batch.grad
            if torch.all( torch.abs(dist) < precision): break
            if normalize_grad:
                gnorm = torch.norm(grad, dim=1).unsqueeze(1)
                grad /= gnorm
            batch = batch - grad * dist
            batch = batch.detach()
        output.append(batch.detach().cpu().numpy())
    return np.concatenate(output)


def gradient_descent(
    query_pts : np.ndarray, 
    model : torch.nn.Module, 
    device : str = "cpu", 
    step_size : float = 1.,
    batch_size : int = 1000, 
    use_dist : bool = False,
    normalize_grad : bool = False,
    max_steps : int = 100
) -> np.ndarray :
    """Performs steps of gradient descent of the given neural implicit function.

    Args:
        query_pts (np.ndarray): starting points.
        model (torch.nn.Module): the neural implicit function.
        device (str, optional): device onto which the computation is performed. Defaults to "cpu".
        step_size (float, optional): multiplicative parameter of the gradient at each step ("learning rate"). Defaults to 1..
        batch_size (int, optional): batch size for forward/backward computation on the neural implicit. Defaults to 1000.
        use_dist (bool, optional): whether to multiply the step by the value of the function at each point. Defaults to False.
        normalize_grad (bool, optional): whether to normalize the neural implicit gradient at each steps. Defaults to False.
        max_steps (int, optional): maximum number of steps. Defaults to 100.

    Returns:
        np.ndarray: output points
    """
    query_tnsr = torch.Tensor(query_pts).to(device)
    loader = DataLoader(query_tnsr, batch_size=batch_size)
    output = []
    for batch in loader:
        for step in range(max_steps):
            batch.requires_grad = True
            dist = model(batch)
            torch.sum(dist).backward()
            grad = batch.grad
            if normalize_grad:
                grad = grad / torch.norm(grad, dim=1)
            if use_dist:
                batch = batch - step_size*dist*grad
            else:
                batch = batch - step_size*grad
            batch = batch.detach()
        output.append(batch.detach().cpu().numpy())
    return np.concatenate(output)


def sample_iso_projection(
    init_pts : np.ndarray, 
    model : torch.nn.Module, 
    iso : float = 0.,
    device : str = "cpu",
    n_nearest : int = 10,
    stdv : float = 0.2,
    step_size : float = 0.1, 
    batch_size : int = 1000, 
    normalize_grad : bool=False, 
    max_steps : int = 10
) -> np.ndarray:
    """
    Sample a neural implicit isosurface by projecting points onto it and make them repel themselves.

    Args:
        init_pts (np.ndarray): initial points to be projected onto the isosurface.
        model (torch.nn.Module): the neural implicit function.
        iso (float, optional): which isovalue the points should be projected to. Defaults to 0..
        device (str, optional): device onto which the computation is performed. Defaults to "cpu".
        n_nearest (int, optional): number of nearest neighbors to consider for the repelling phase. Defaults to 10.
        stdv (float, optional): standard deviation of the weights of near samples. Defaults to 0.2.
        step_size (float, optional): multiplicative parameter of the gradient at each step ("learning rate"). Defaults to 1..
        batch_size (int, optional): batch size for forward/backward computation on the neural implicit. Defaults to 1000.
        normalize_grad (bool, optional): whether to normalize the neural implicit gradient at each steps. Defaults to False.
        max_steps (int, optional): maximum number of steps. Defaults to 100.

    Returns:
        np.ndarray: points sampled on the given isosurface.
    
    References:
        Iso-Points: Optimizing Neural Implicit Surfaces with Hybrid Representations, Yifan et al. (2021)
    """
    samples = project_onto_iso(init_pts, model, iso=iso, device=device, batch_size=batch_size, normalize_grad=normalize_grad, max_steps=100)  
    for step in range(max_steps):
        print(step)

        ### Compute KDtree and make point repulse themselves
        kdtree = KDTree(samples)
        dist_nn, ind_nn = kdtree.query(samples, k=n_nearest+1)
        dist_nn, ind_nn = dist_nn[:,1:], ind_nn[:,1:] # discard self
        k_nn = samples[ind_nn]
        directions = k_nn - samples[:,None,:]

        # project into local tangent plane
        # normals = forward_in_batches(model,samples,device,compute_grad=True, batch_size=batch_size)[1]
        # normals /= np.linalg.norm(normals,axis=1)[:,np.newaxis]
        # directions = directions - np.dot(directions,normals) * normals

        # move the samples
        directions /= np.linalg.norm(directions,axis=2)[:,:,np.newaxis]
        weights = np.exp(-dist_nn**2/stdv)[:,:,None]
        samples = samples - step_size * np.sum(weights * directions,axis=1)

        ### Project back onto surface
        samples = project_onto_iso(samples, model, iso=iso, device=device, batch_size=batch_size, normalize_grad=normalize_grad, max_steps=10)
    return samples



    
class _RayTracingImplicitSampler:
    def __init__(self, model : torch.nn.Module, device : str, iso : float, thresh : float = 1e-6): 
        self.model = model
        self.device : str = device
        self.threshold = thresh
        self.step_bound = 1.
        self.iso = iso
    
    @property
    def dim(self) -> int:
        return self.model.dim
    
    def sdf_fn(self, x):
        return self.model(x) - self.iso

    def sphere_trace_modified(self, origin, direction, max_t):
        all_pts = []
        all_pts_hit = []
        sdf_cost = 0
        
        p = origin + 0*direction
        d = torch.squeeze(self.sdf_fn(p))
        t = torch.zeros(d.shape).to(origin)
        
        delta = torch.abs(d)
        not_converged = (t < max_t) # march every ray till the end
        for iter in trange(2000):
            if not torch.any(not_converged): break 
            
            # add samples near surface and step over the surface for these samples 
            p_near_surface = (t < max_t) & (delta < self.threshold)
            if torch.any(p_near_surface):
                p_hits = torch.ones_like(p).to(p)
                p_hits[p_near_surface] = p[p_near_surface]
                all_pts.append(p_hits)
                all_pts_hit.append(p_near_surface)
                
                # gather samples near surface 
                p_near = p[p_near_surface]
                delta_near = delta[p_near_surface]
                t_near = t[p_near_surface]
                max_t_near = max_t[p_near_surface]
                directions_near = direction[p_near_surface]
                
                # p_near_before = p_near.clone()
                # Keep stepping until delta > eps
                not_flipped = (t_near < max_t_near) & (delta_near < self.threshold)
                while torch.any(not_flipped):
                
                    # Take a step for the samples not flipped yet 
                    p_near_not_flipped = p_near[not_flipped]
                    delta_near_not_flipped = delta_near[not_flipped]
                    t_near_not_flipped = t_near[not_flipped]
                    
                    # Take a step of size max( delta_near_not_flipped, eps/2 ) so that it doesn't get stuck with too small a step.
                    p_near_not_flipped = p_near_not_flipped + torch.maximum(delta_near_not_flipped[:,None], torch.ones_like(delta_near_not_flipped)[:,None] * self.threshold/2.0) * directions_near[not_flipped]
                    t_near_not_flipped = t_near_not_flipped + torch.maximum(delta_near_not_flipped, torch.ones_like(delta_near_not_flipped) * self.threshold/2.0)
            
                    # Measure distance at new location
                    with torch.no_grad():
                        delta_near_not_flipped = torch.squeeze(self.sdf_fn(p_near_not_flipped))
                    sdf_cost += p_near_not_flipped.shape[0]
                    delta_near_not_flipped = torch.abs(delta_near_not_flipped)
                    
                    # Copy back to these near surface samples 
                    p_near[not_flipped] = p_near_not_flipped
                    t_near[not_flipped] = t_near_not_flipped
                    delta_near[not_flipped] = delta_near_not_flipped
                    
                    # Evaluate whether every sample is flipped 
                    not_flipped = (t_near < max_t_near) & (delta_near < self.threshold)
                
                # after flipping these samples to the other side of the surface, put those samples back with updated t and delta 
                p[p_near_surface] = p_near
                delta[p_near_surface] = delta_near
                t[p_near_surface] = t_near
            
            nc_far = not_converged
            p_far = p[nc_far,:]
            d_far = d[nc_far]
            delta_far = delta[nc_far]
            t_far = t[nc_far]

            # Take a step
            p_far = p_far + delta_far[:,None]/self.step_bound * direction[nc_far,:]
            t_far = t_far + delta_far/self.step_bound

            # Measure distance at new location
            with torch.no_grad():
                d_far = torch.squeeze(self.sdf_fn(p_far))
                
            sdf_cost += p_far.shape[0]
            delta_far = torch.abs(d_far)

            # Copy back to original tensors
            p[nc_far,:] = p_far
            d[nc_far] = d_far
            delta[nc_far] = delta_far
            t[nc_far] = t_far

            iter += 1
            not_converged = t < max_t
            
        if len(all_pts) > 0:
            all_pts = torch.stack(all_pts, dim=-2)
            all_pts_hit = torch.stack(all_pts_hit, dim=-1)
            pts = all_pts.view(-1,self.dim)[all_pts_hit.view(-1)]
            return pts, sdf_cost
        else: 
            return torch.zeros(0,self.dim), sdf_cost
    

    def sample_rays(self, num_rays):
        # sample line directions 
        dirs = np.random.randn(num_rays, self.dim)
        dirs = dirs / np.linalg.norm(dirs,axis=-1)[:,None] #[n,dim]
        dirs = dirs[:,None,:] #[n,1,3]
        
        # find the other normal and binormal basis 
        _,_,V = np.linalg.svd(dirs) 
        E = V[:,:,1:] #[n,3,2] 
        
        # sample random offsets in normal and binormal directions
        dirs = dirs[:,0,:] 
        U = np.sqrt(self.dim) * ( np.random.uniform(0,1,(num_rays,1, self.dim-1)) * 2.2 - 1.1)  
        O = np.sum(U * E, axis=-1) 
        O = O + dirs * np.sqrt( self.dim)
        
        # determine if the ray O+D*t  intersects the [-1.1, 1.1]^dim hypercube
        # using Slab method
        t_low = np.zeros((num_rays, self.dim))
        t_high = np.zeros((num_rays, self.dim))
        t_low = (-1.1 - O)/dirs 
        t_high = (1.1 - O)/dirs
        
        t_close = np.minimum(t_low, t_high)
        t_far = np.maximum(t_low,t_high)
        t_close = np.max(t_close, axis=-1)
        t_far = np.min(t_far,axis=-1)
        keep = t_close < t_far
        
        dirs = dirs[keep]
        O = O[keep]
        t_close = t_close[keep]
        t_far = t_far[keep]
        O = O + dirs * t_close[:,None]
        T = t_far - t_close 
        n_current = O.shape[0]
        
        if (not np.any(keep)) and n_current > num_rays:
            return np.zeros((0,self.dim)),np.zeros((0,self.dim)),np.zeros((0))
        elif n_current == num_rays:
            return O, dirs, T
        elif n_current < num_rays: 
            O_current, D_current,T_current = self.sample_rays(num_rays-n_current)
            O = np.concatenate([O,O_current],axis=0)
            D = np.concatenate([dirs,D_current],axis=0)
            T = np.concatenate([T,T_current],axis=0)
            return O, D, T 
    
    def sample(self, num_rays : int):
        lines_origins, lines_dirs, max_ts = self.sample_rays(num_rays)     
        # currently assume everything runs with pytorch on cuda 
        lines_origins = torch.from_numpy(lines_origins).to(torch.float32).to(self.device)
        lines_dirs = torch.from_numpy(lines_dirs).to(torch.float32).to(self.device)
        pts, _ = self.sphere_trace_modified(
            lines_origins, 
            lines_dirs,
            max_t=torch.from_numpy(max_ts).to(torch.float32).to(self.device),
        ) 
        return pts 

def sample_iso_raytraced(
    model : torch.nn.Module,
    n_rays : int,
    iso : float = 0.,
    device : str = "cpu",
    threshold : float = 1e-4,
) -> np.ndarray:
    """
    Sample an isosurface of a neural implicit function by tracing rays intersecting the object and returning all ray-shape intersection points.

    Args:
        model (torch.nn.Module): the neural implicit function.
        n_rays (int): number of rays to sample. The number of sampled points is related to the number of rays
        iso (float, optional): _description_. Defaults to 0..
        device (str, optional): _description_. Defaults to "cpu".
        threshold (float, optional): _description_. Defaults to 1e-4.

    Returns:
        np.ndarray: sampled points

    References:
        - Uniform Sampling of Surfaces by Casting Rays, Ling et al., 2025  
        - https://github.com/iszihan/implicit-uniform-sampler/blob/main/ImplicitUniformSampler/sampler.py  
    """
    sampler = _RayTracingImplicitSampler(model, device, iso, threshold)
    return sampler.sample(n_rays)


def sample_skeleton(
    model : torch.nn.Module, 
    res : int, 
    device : str = "cpu", 
    batch_size : int = 1000, 
    margin : float =-1e-2, 
    grad_norm_threshold : float = 0.5, 
    descent_steps : int = 0 
) -> np.ndarray:
    """Samples the skeleton of a neural signed distance field. The skeleton of a shape is defined as the inside discontinuities of its distance field's gradient. This function works by sampling and rejecting points depending on the norm of the gradient at each point. If the neural implicit is well behaved (for instance, a 1-Lip SDF), then the discontinuities have low gradient.

    Args:
        model (torch.nn.Module): the neural implicit function.
        res (int): resolution of the sampling.
        device (str, optional): device onto which the computation is performed. Defaults to "cpu".
        batch_size (int, optional): batch size for forward/backward computation on the neural implicit. Defaults to 1000.
        margin (float, optional): the maximal value of the implicit function for a point to be considered inside the shape. Defaults to -1e-2.
        grad_norm_threshold (float, optional): the threshold for rejecting non-skeleton points. Defaults to 0.5.
        descent_steps (int, optional): number of gradient descent step to perform on the sampled points to project them to the skeleton. Defaults to 0.

    Returns:
        np.ndarray: sampled points on the skeleton of the shape
    """
    L = [np.linspace(-1., 1., res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    distances = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    distances = np.squeeze(distances)
    pts = pts[distances<margin,:]
    _,grad = forward_in_batches(model, pts, device, compute_grad=True, batch_size=batch_size)
    grad_norm = np.linalg.norm(grad,axis=1)
    pts = pts[grad_norm<grad_norm_threshold]

    if descent_steps>0:
        pts = gradient_descent(pts, model, device, step_size=1e-2, batch_size=batch_size, max_steps=descent_steps)
    return pts