import mouette as M
import numpy as np
import igl
from scipy.spatial import KDTree
import triangle
from collections import deque
from tqdm import trange

from base import FieldGenerator, UnsupportedDimensionError, UnsupportedGeometryFormat

#######################################################################################

def WindingNumber(dim, geom : M.mesh.Mesh):
    if dim!=3: raise UnsupportedDimensionError(dim)
    
    if isinstance(geom, M.mesh.SurfaceMesh):
        return _WindingNumber3D_TriangleSoup(dim, geom)
    elif isinstance(geom, M.mesh.PointCloud):
        return _WindingNumber3D_PointCloud(dim, geom)
    raise UnsupportedGeometryFormat(geom)


#######################################################################################

class _WindingNumber3D_TriangleSoup(FieldGenerator):
    
    def __init__(self, mesh : M.mesh.SurfaceMesh):
        super().__init__()
        self.V = np.asarray(mesh.vertices)
        self.F = np.asarray(mesh.faces)

    def compute(self, query: np.ndarray) -> np.ndarray:
        return igl.fast_winding_number(self.V, self.F, query)

#######################################################################################

class _WindingNumber3D_PointCloud(FieldGenerator):
    
    def __init__(self, pc : M.mesh.PointCloud):
        super().__init__()
        self.pc = pc
        self.V = np.asarray(pc.vertices)
        
        if not pc.vertices.has_attribute("normals") or not pc.vertices.has_attribute("area"):
            # precompute kdtree once
            kdtree = KDTree(pc.vertices)

        if pc.vertices.has_attribute("normals"):
            self.N = pc.vertices.get_attribute("normals").as_array(len(pc.vertices))
        else:
            print("Estimate normals")
            self.N = self._estimate_normals(self.V, kdtree)
            pc.vertices.register_array_as_attribute("normals", self.N)

        if pc.vertices.has_attribute("area"):
            self.A = pc.vertices.get_attribute("area").as_array(len(pc.vertices))
        else:
            print("Estimate vertex local area")
            self.A = self._estimate_vertex_areas(self.V, self.N, kdtree)
            pc.vertices.register_array_as_attribute("area", self.A)


    def compute(self, query: np.ndarray) -> np.ndarray: 
        return igl.fast_winding_number_for_points(self.V, self.N, self.A, query)
    

    def _estimate_normals(self, V, tree:KDTree, k=30):
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

    def _estimate_vertex_areas(self, V, N, tree:KDTree, k=10):
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