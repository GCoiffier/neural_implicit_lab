import implicitlab as IL
import polyscope as ps
import sys
import torch

DEVICE = IL.utils.get_device()
inr = IL.nn.load_model(sys.argv[1]).to(DEVICE)
inr = torch.compile(inr)

ps.init()
ps.set_ground_plane_mode("none")
torch.set_float32_matmul_precision("high")

def polyscope_wrapper(pts):
    print(pts.shape)
    pts_torch = torch.tensor(pts, device=DEVICE)
    values = IL.utils.forward_in_batches(inr, pts_torch, device=DEVICE)
    values = values.squeeze(1).cpu().numpy()
    return values

ps.render_implicit_surface("neural_sdf", polyscope_wrapper, 
    mode='sphere_march', enabled=True)

ps.show()