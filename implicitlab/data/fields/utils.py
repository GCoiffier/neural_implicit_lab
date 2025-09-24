import mouette as M
import numpy as np

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