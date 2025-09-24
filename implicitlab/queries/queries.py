import numpy as np
from .sdf import SignedDistanceField
import mouette as M
from skimage.measure import marching_cubes

class ImplicitQueries:

    def __init__(self, SDF: SignedDistanceField, precision:float = 1e-8, infty:float = 10.):
        self.sdf = SDF
        self.precision : float = precision
        self.infty : float = infty


    def query(self, pos):
        return self.sdf(pos)


    def project_to_iso(self, pos, iso:float = 0., max_steps:int = 100, normalize_grad:bool = False):
        pos = np.copy(pos)
        for _ in range(max_steps):
            dist, grad = self.sdf.fungrad(pos)
            converged = np.abs(dist-iso) < self.precision 
            if np.all(converged): break
            if normalize_grad:
                gnorm = np.linalg.norm(grad, axis=1)[:, np.newaxis]
                grad /= gnorm
            pos = np.where(~converged, pos - grad * dist, pos)
        return pos


    def ray_cast(self, pos, dir, max_steps:int = 100):
        pos = np.copy(pos)
        converged = np.zeros(pos.shape[0], dtype=bool)
        for _ in range(max_steps):
            dists = self.query(pos)
            converged = (np.abs(dists)<self.precision) | (np.abs(dists)>self.infty)
            if np.all(converged):
                return pos
            pos = np.where(~converged, pos + dir*dists, pos)
        return pos, (np.abs(dists)<self.precision)
    
    
    def reconstruct_surface_marching_cubes(self, iso:float=0., res:int = 100):
        padding = 0.05*np.min(self.sdf.bb.span)
        L = [np.linspace(self.sdf.bb.mini[i] - padding, self.sdf.bb.maxi[i] + padding, res) for i in range(3)]
        pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
        dist_values = self.query(pts)
        dist_values = dist_values.reshape((res,res,res))

        ### Call marching cubes
        verts,faces,normals,values = marching_cubes(dist_values, level=iso)
        values = values[:, np.newaxis]
        mc_mesh = M.mesh.RawMeshData()
        mc_mesh.vertices += list(verts)
        mc_mesh.faces += list(faces)
        mc_mesh = M.mesh.SurfaceMesh(mc_mesh)
        mc_mesh.vertices.register_array_as_attribute("normals", normals)
        mc_mesh.vertices.register_array_as_attribute("values", values)
        
        ### Reproject meshes to correct coordinates
        for v in mc_mesh.id_vertices:
            pV = M.Vec(mc_mesh.vertices[v])
            ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
            dx, dy, dz = pV.x%1, pV.y%1, pV.z%1
            vx = (1-dx)*L[0][ix] + dx * L[0][ix+1]
            vy = (1-dy)*L[1][iy] + dy * L[1][iy+1]
            vz = (1-dz)*L[2][iz] + dz * L[2][iz+1]
            mc_mesh.vertices[v] = M.Vec(vx,vy,vz)
        return mc_mesh