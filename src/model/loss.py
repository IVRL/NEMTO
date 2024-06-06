import torch
from torch import nn
from torch.nn import functional as F

class IDRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, alpha, r_patch, rgb_weight, normalsmooth_weight, refraction_weight, refraction_smooth_weight=0):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.refraction_smooth_weight = refraction_smooth_weight
        self.r_patch = int(r_patch)
        self.rgb_weight = rgb_weight
        self.refraction_weight = refraction_weight
        self.normalsmooth_weight = normalsmooth_weight

        print('Patch size in normal smooth loss: ', self.r_patch)

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, mask_sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * mask_sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_dir_loss(self, pred_dirs, calc_dirs, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        pred_dirs = pred_dirs.reshape(-1, 3)[network_object_mask & object_mask]
        calc_dirs= calc_dirs.reshape(-1, 3)[network_object_mask & object_mask]
        assert pred_dirs.shape[0] == calc_dirs.shape[0]

        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos_sim(pred_dirs, calc_dirs).mean()

        return 1 - sim


    def get_smooth_loss(self, dir, normal, network_object_mask, object_mask):
        mask = (network_object_mask & object_mask).reshape(-1, 4*self.r_patch*self.r_patch).all(dim=-1)
        if self.r_patch < 1 or mask.sum() == 0.:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        dir = dir.view((-1, 4*self.r_patch*self.r_patch, 3))
        normal = normal.view((-1, 4*self.r_patch*self.r_patch, 3))
        return torch.mean(torch.var(dir, dim=1)[mask]), torch.mean(torch.var(normal, dim=1)[mask])

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['mask_sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        refraction_smooth_loss, normalsmooth_loss = self.get_smooth_loss(model_outputs['pred_refraction'], model_outputs['normal_values'], network_object_mask, object_mask)
        refraction_loss = self.get_dir_loss(model_outputs['pred_refraction'], model_outputs['calc_reflection'],  network_object_mask, object_mask)

        loss = self.rgb_weight * rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.refraction_weight * refraction_loss + \
               self.refraction_smooth_weight * refraction_smooth_loss + \
               self.normalsmooth_weight * normalsmooth_loss
               

        return {
            'loss': loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'rgb_loss': rgb_loss,
            'refraction_loss': refraction_loss,
            'refraction_smooth_loss': refraction_smooth_loss, 
            'normalsmooth_loss': normalsmooth_loss
        }
