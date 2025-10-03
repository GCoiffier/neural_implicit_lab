import mouette as M
import numpy as np
import triangle
from collections import deque
from tqdm import trange
from scipy.spatial import KDTree

def pseudo_surface_from_polyline(pl):
    """The signed distance function of libigl only works in 3D. Since I am too lazy to code an equivalent function for 2D inputs, this function transforms the 2D polyline into a 3D surface by linking each edge (a,b) to two far away points Z+ and Z- to form two triangles.

    Args:
        pl (M.mesh.PolyLine): input polyline

    Returns:
        np.ndarray, np.ndarray: Extruded surface as V,F where V is a numpy array of points of shape (N+2,3) and F is a (M,3) numpy array of face indices
    """
    nv = len(pl.vertices)
    bary = M.attributes.barycenter(pl)
    bary1 = M.Vec(bary.x, bary.y, 100)
    bary2 = M.Vec(bary.x, bary.y, -100)
    V = list(pl.vertices)+[bary1,bary2]
    F = []
    for a,b in pl.edges:
        if a==0 and b==len(pl.edges)-1:
            ### last edge (n-1, 0) is oriented backwards
            F += [(b,a,nv), (a,b,nv+1)]
        else:
            F += [(a,b,nv), (b,a,nv+1)]
    ### Correctly orient normals outward
    ref = M.Vec(-1., 0., 0.) # point outside of the mesh
    volume = 0.
    for (A,B,C) in F:
        pA,pB,pC = V[A], V[B], V[C]
        volume += M.geometry.det_3x3(pB-pA, ref-pA, pC-pA)
    if volume<0:
        F = [(b,a,c) for (a,b,c) in F] # change order of faces
    return np.array(V), np.array(F, dtype=int)


def estimate_normals(V, tree:KDTree, k=30):
    """Estimation of outward normals for each vertex of the point cloud. Computed as a linear regression on k nearest neighbors

    Args:
        V (np.ndarray): array of vertex positions (Nx3)
        tree (KDTree): kdtree for fast neighbor queries
        k (int, optional): number of neighbor vertices to take into account for computation. Defaults to 30.

    Returns:
        np.ndarray: array of estimated normals (Nx3)
    """
    n_pts = V.shape[0]
    _, KNN = tree.query(V, k+1)
    KNN = KNN[:,1:] # remove self from nearest
    
    # Compute normals
    N = np.zeros((n_pts, 3))
    for i in trange(n_pts):
        mat_i = np.array([V[nn,:]-V[i,:] for nn in KNN[i,:]])
        mat_i = mat_i.T @ mat_i
        _, eigs = np.linalg.eigh(mat_i)
        N[i,:] = eigs[:,0]

    # Consistent normal orientation
    visited = np.zeros(n_pts,dtype=bool)
    to_visit = deque()
    exterior_pt = M.Vec(10,0,0)
    _,top_pt = tree.query(exterior_pt)
    if np.dot(N[i,:], exterior_pt - V[top_pt,:])<0.:
        N[i,:] *= -1 # if this normal is not outward, we are doomed
    to_visit.append(top_pt)
    while len(to_visit)>0:
        iv = to_visit.popleft()
        if visited[iv] : continue
        visited[iv] = True
        Nv = N[iv, :]
        for nn in KNN[iv,:]:
            if np.dot(Nv, N[nn,:])<0.:
                N[nn,:] *= -1
            to_visit.append(nn)
    return N


def estimate_vertex_areas(V, N, tree:KDTree, k=10):
    """The generalized winding number needs an estimate of 'local areas' of each points. This functions computes this area estimation as described in the article of Barill et al. (2018).

    Args:
        V (np.ndarray): array of vertex positions (Nx3)
        N (np.ndarray): array of normal vector per vertex (Nx3)
        tree (KDTree): kdtree for fast neighbor queries
        k (int, optional): number of neighbor vertices to take into account for computation. Defaults to 20.

    Returns:
        np.ndarray: array of estimated local areas (Nx1)
    """
    n_pts = V.shape[0]
    _, KNN = tree.query(V,k+1)
    KNN = KNN[:,1:]
    A = np.zeros(n_pts)
    for i in trange(n_pts):
        ni = M.Vec.normalized(N[i])
        Xi,Yi,Zi = (M.geometry.cross(basis, ni) for basis in (M.Vec.X(), M.Vec.Y(), M.Vec.Z()))
        Xi = [_X for _X in (Xi,Yi,Zi) if M.geometry.norm(_X)>1e-8][0]
        Xi = M.Vec.normalized(Xi)
        Yi = M.geometry.cross(ni,Xi)

        neighbors = [V[j] for j in KNN[i,:]] # coordinates of k neighbors
        neighbors = [M.geometry.project_to_plane(_X,N[i],V[i]) for _X in neighbors] # Project onto normal plane
        neighbors = [M.Vec(Xi.dot(_X), Yi.dot(_X)) for _X in neighbors]

        for (p1,p2,p3) in triangle.triangulate({"vertices" : neighbors})["triangles"]:
            A[i] += M.geometry.triangle_area_2D(neighbors[p1], neighbors[p2], neighbors[p3])
    return A