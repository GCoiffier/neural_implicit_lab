import implicitlab as IL
import mouette as M
import sys

DEVICE = IL.utils.get_device()
model = IL.nn.load_model(sys.argv[1], DEVICE)

domain = M.geometry.AABB([-1.2]*3, [1.2]*3)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE)
for name, iso in data.items():
    M.mesh.save(iso, "iso.obj")

    