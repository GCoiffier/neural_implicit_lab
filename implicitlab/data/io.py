import mouette as M
import meshio

def load_geometry(file_path : str):
    try:
        raise NotImplementedError
        mesh = meshio.read(path)
    except:
        mesh = M.mesh.load(file_path)
    mesh = M.transform.normalize(mesh)
    if isinstance(mesh, M.mesh.SurfaceMesh):
        mesh = M.mesh.triangulate(mesh)
    return mesh