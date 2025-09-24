import mouette as M
import numpy as np

def is_normalized(mesh : M.mesh.Mesh) -> bool:
    coords = np.asarray(mesh.vertices)
    return np.min(coords) >= -1 and np.max(coords) <= 1. 

def get_data_dimensionnality(mesh : M.mesh.Mesh) -> int:
    mean_dim = np.mean(mesh.vertices,axis=0)
    if mean_dim.size==2: return 2
    return np.sum(abs(mean_dim)>1e-8)

