import implicitlab.utils as Imp
import mouette as M
import os
import numpy as np

os.makedirs("output", exist_ok=True)

spot_sdf = Imp.NeuralDistanceField("inputs/spot.pt", force_cpu=False, batch_size=10_000)
spot = Imp.ImplicitQueries(spot_sdf, precision=1e-6, infty=2.)


print("marching cubes")
if not os.path.exists("output/spot.obj"):
    M.mesh.save(spot.reconstruct_surface_marching_cubes(res=200), "output/spot.obj")


print("project on surface")
sphere = M.sampling.sample_sphere(M.Vec.zeros(3), 1. , n_pts=1000)
project = spot.project_to_iso(sphere, normalize_grad=True)
M.mesh.save(M.procedural.vector_field(sphere, project-sphere), "output/projection.mesh")


print("ray casting")
init_quad = M.procedural.quad(M.Vec(-0.6,-0.6, -0.6), M.Vec(-0.6, 0.6,-0.6), M.Vec(-0.6, -0.6, 0.6), triangulate=True)
pos = M.sampling.sample_surface(init_quad, 500)
ray_marched, converged = spot.ray_cast(pos, M.Vec(1,0,0))

output = M.procedural.vector_field(pos, ray_marched-pos)
output.vertices.register_array_as_attribute("converged", np.repeat(converged,2))
M.mesh.save(output, "output/ray_marched.geogram_ascii")
M.mesh.save(output, "output/ray_marched.mesh")
