import argparse
import implicitlab as IL
import mouette as M
import torch
import warnings

parser = argparse.ArgumentParser("Implicitlab: Mesh extraction from a neural implicit using the marching cube algorithm")

parser.add_argument("inr", type=str, help="path to the implicit neural representation file to be loaded")
parser.add_argument("-res", type=int, default=512, help="Resolution of the marching cubes algorithm. Defaults to 512^3")
parser.add_argument("--batch-size", type=int, default=2**16, help="model's batch size")
parser.add_argument("--output-name", type=str, default="iso")
args = parser.parse_args()

DEVICE = IL.utils.get_device()
if not torch.cuda.is_available():
    warnings.warn("Pytorch does not have access to cuda and falls back to CPU mode. Forward computation of the grid values may take a long time.")

model = IL.nn.load_model(args.inr).to(DEVICE)
model = model.eval()
domain = M.geometry.AABB([-1.1]*3, [1.1]*3)
data = IL.visualize.reconstruct_surface_marching_cubes(model, domain, DEVICE, res=512, batch_size=2**16, use_tqdm=True, full_precision=True)
for name, iso in data.items():
    M.mesh.save(iso, f"{args.output_name}.obj")

    