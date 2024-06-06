import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
import utils.plots as plt
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.ior_wt import RefractionDirNetwork
from model.rbn import render_glass
from model.rbn import network_transmission

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        self.feature_vector_size = feature_vector_size
        print('ImplicitNetowork feature_vector_size: ', self.feature_vector_size)
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, 3)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            ## initiation such that the implicit sdf is roughly a sphere, citing igr
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class TransparentNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.implicit_network = ImplicitNetwork(**conf.get_config('implicit_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.refraction_net = RefractionDirNetwork(**conf.get_config('refraction_network'))

    def freeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = False
        
    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        envmap = input["envmap"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        mask_sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = mask_sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        normal_values = torch.ones_like(points).float().cuda()
        rgb_values = torch.ones_like(points).float().cuda()
        pred_refraction_dirs = torch.ones_like(points).float().cuda()
        calc_reflection_dirs = torch.ones_like(points).float().cuda()


        if differentiable_surface_points.shape[0] > 0:
            view_dirs = -ray_dirs[surface_mask]  # ----> camera
            # sdf = self.implicit_network(points)
            ret = self.get_refraction(differentiable_surface_points, view_dirs, envmap, self.implicit_network)
            normal_values[surface_mask] = ret['normals']
            rgb_values[surface_mask] = ret['rgb_vals']
            pred_refraction_dirs[surface_mask] = ret['predict_refraction']
            calc_reflection_dirs[surface_mask] = ret['gt_refraction']


        output = {
            'points': points,
            'mask_sdf_output': mask_sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'normal_values': normal_values,
            'rgb_values': rgb_values,
            'pred_refraction': pred_refraction_dirs,
            'calc_reflection': calc_reflection_dirs,
        }

        return output


    def get_refraction(self, diff_points, view_dirs, envmap, sdf):
        g = self.implicit_network.gradient(diff_points)
        normals = g[:, 0, :]
        normals= normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        ret = { 'normals': normals, }

        refraction_dict = self.refraction_net(torch.cat([diff_points, normals, view_dirs], -1))

        # with torch.no_grad():
        calculated_refraction = network_transmission(sdf, diff_points, normals, -view_dirs, refraction_dict['ior']).detach()

        ret["gt_refraction"] = calculated_refraction

        rgb_ret = render_glass(refraction_dict['ior'],
                               refraction_dict['refraction_direction'],
                               normals,
                               view_dirs,
                               envmap)

        ret['predict_refraction'] = refraction_dict['refraction_direction']

        ret.update(rgb_ret)
        return ret







