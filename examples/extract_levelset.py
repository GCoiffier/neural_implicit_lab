import argparse
import implicitlab as IL
import mouette as M
import sys
import torch

DEVICE = IL.utils.get_device()
model = IL.nn.load_model(sys.argv[1]).to(DEVICE)
model = model.eval()
domain = M.geometry.AABB([-1.]*3, [1.]*3)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, res=512, batch_size=2**16, use_tqdm=True)
for name, iso in data.items():
    M.mesh.save(iso, "iso.obj")

    