import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util

def plot(model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, plot_nimgs, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)

    depth = torch.ones(batch_size * num_samples).cuda().float()
    if network_object_mask.sum() > 0:
        depth_valid = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
        depth[network_object_mask] = depth_valid
        depth[~network_object_mask] = 0.98 * depth_valid.min()
    depth = depth.reshape(batch_size, num_samples, 1)
    # plot rendered images
    plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    plot_depth_maps(depth, indices, path, epoch, plot_nimgs, img_res)


def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.symeig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, indices, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}_{2}.png'.format(path, epoch, indices))


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = ground_true.cuda()
    gamma = 1.0

    tonemap_img = lambda x: torch.pow(x, 1./gamma)
    clip_img = lambda x: torch.clamp(x, min=0., max=1.)

    rgb_points = clip_img(tonemap_img(rgb_points))
    

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))
