import numpy as np
import mouette as M
import torch
from skimage.measure import marching_cubes

from .utils import forward_in_batches

def parameter_singular_values(model):
    layers = list(model.children())
    data= []
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.linalg.svd(w)
            # data.append(f"{layer}, {s}")
            data.append(f"{layer}, min={s.min()}, max={s.max()}")
    return data


def reconstruct_surface_marching_cubes(model, domain, device, iso=0, res=100, batch_size=5000):
    if isinstance(iso, (int,float)): iso = [iso]
    
    ### Feed grid to model
    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.reshape((res,res,res))

    ### Call marching cubes
    to_save = dict()
    for ioff,off in enumerate(iso):
        try:
            verts,faces,normals,values = marching_cubes(dist_values, level=off)
            values = values[:, np.newaxis]
            m = M.mesh.RawMeshData()
            m.vertices += list(verts)
            m.faces += list(faces)
            m = M.mesh.SurfaceMesh(m)
            normal_attr = m.vertices.create_attribute("normals", float, 3, dense=True)
            normal_attr._data = normals
            values_attr = m.vertices.create_attribute("values", float, 1, dense=True)
            values_attr._data = values
            to_save[(ioff, off)] = m
        except ValueError:
            continue
    
    ### Reproject meshes to correct coordinates
    for key,mesh in to_save.items():
        for v in mesh.id_vertices:
            pV = M.Vec(mesh.vertices[v])
            ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
            dx, dy, dz = pV.x%1, pV.y%1, pV.z%1
            vx = (1-dx)*L[0][ix] + dx * L[0][ix+1]
            vy = (1-dy)*L[1][iy] + dy * L[1][iy+1]
            vz = (1-dz)*L[2][iz] + dz * L[2][iz+1]
            mesh.vertices[v] = M.Vec(vx,vy,vz)
        to_save[key] = mesh
    return to_save