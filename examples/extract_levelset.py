import implicitlab as IL
import mouette as M
import sys
import torch

DEVICE = IL.utils.get_device()
model = IL.nn.load_model(sys.argv[1], DEVICE)
# model = IL.nn.SirenNet(3, 256, 6).to(DEVICE)
# model.load_state_dict(torch.load(sys.argv[1]))

domain = M.geometry.AABB([-1.]*3, [1.]*3)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, res=500)
for name, iso in data.items():
    M.mesh.save(iso, "iso.obj")

    