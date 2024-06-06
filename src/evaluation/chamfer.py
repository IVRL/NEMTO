import argparse

import torch
import numpy as np

# pip install trimesh[all]
import trimesh

# https://github.com/otaheri/chamfer_distance
from chamfer_distance import ChamferDistance

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")

if __name__ == "__main__":
    chamfer_dist = ChamferDistance()

    parser = argparse.ArgumentParser(description='Chamfer loss')
    parser.add_argument('-n', type=int, default=2500000)
    parser.add_argument('mesh', type=str)
    parser.add_argument('ref', type=str)
    FLAGS = parser.parse_args()

    print('Loading geometry ...')
    mesh = as_mesh(trimesh.load(FLAGS.mesh))
    ref = as_mesh(trimesh.load(FLAGS.ref))

    # Make sure l=1.0 maps to 1/10th of the AABB. https://arxiv.org/pdf/1612.00603.pdf
    scale = 10.0 / np.amax(np.amax(ref.vertices, axis=0) - np.amin(ref.vertices, axis=0))
    ref.vertices = ref.vertices * scale
    mesh.vertices = mesh.vertices * scale

    # Sample mesh surfaces
    print('Sampling mesh points ...')
    vpos_mesh = sample_mesh(mesh, FLAGS.n)
    vpos_ref = sample_mesh(ref, FLAGS.n)

    dist1, dist2, _, _ = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    
    print("[%7d tris]: %1.5f" % (mesh.faces.shape[0], loss))