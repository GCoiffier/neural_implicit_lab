import numpy as np
import mouette as M

def sample_points_and_normals2D(polyline, n_pts):
    sampled_pts = np.zeros((n_pts, 2))
    lengths = M.attributes.edge_length(polyline, persistent=False).as_array()
    lengths /= np.sum(lengths)
    if len(polyline.edges)==1:
        edges = [0]*n_pts
    else:
        edges = np.random.choice(len(polyline.edges), size=n_pts, p=lengths)
    sampled_normals = np.zeros((n_pts,2))
    for i,e in enumerate(edges):
        A,B = polyline.edges[e]
        pA,pB = polyline.vertices[A], polyline.vertices[B]
        ni = M.Vec.normalized(pB - pA)
        if B==len(polyline.vertices)-1:
            # the last edge given in order is (n-1, 0) which has been swapped due to mouette convention to (0,n-1), meaning that it is inverted
            ni *= -1
        sampled_normals[i,:] = np.array([-ni.y, ni.x])
        t = np.random.random()
        pt = t*pA + (1-t)*pB
        sampled_pts[i,:] = pt[:2]

    # check normal orientation
    n_view_pts = 21
    view_pts = [10*M.Vec(np.cos(2*np.pi*a), np.sin(2*np.pi*a)) for a in np.random.rand(n_view_pts)]
    sign_count = 0
    for i in range(n_view_pts):
        ind_nn = np.argmin([M.geometry.distance(view_pts[i], _q) for _q in sampled_pts])
        sign_count += M.geometry.sign(np.dot(sampled_normals[ind_nn], view_pts[i] - sampled_pts[ind_nn, :]))
    if sign_count<0:
        sampled_normals *=-1
    return sampled_pts, sampled_normals


def sample_unit_disk(n_pts):
    r = np.sqrt(np.random.random(n_pts))
    theta = 2*np.pi*np.random.random(n_pts)
    pts = np.vstack([r*np.cos(theta), r*np.sin(theta)]).T
    return pts