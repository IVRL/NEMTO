import sys
sys.path.append('../pipeline')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils import vis_util
import imageio.v2 as imageio


def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + '-' + kwargs['expname']


    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER1')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER2')
            exit()
    else:
        timestamp = kwargs['timestamp']
    
    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname, os.path.basename((kwargs['data_split_dir'])))
    if not os.path.exists(evaldir):
        os.makedirs(evaldir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(kwargs['data_split_dir'])                                                                    

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=eval_dataset.collate_fn
                                                  )
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    print('Loaded checkpoint: ', ckpt_path)

    relight = False
    if kwargs['envmap_path'].endswith('.exr'):
        print('Loading light from: ', kwargs['envmap_path'])
        img = imageio.imread(kwargs['envmap_path'])[:, :, :3]
        img = np.float32(img)
        # if not path.endswith('.exr'):
        #     img = img / 255.
        gamma = 2.2
        tonemap_img = lambda x: np.power(x, 1./gamma)
        new_envmap = torch.from_numpy(tonemap_img(img)).float()
        relight = True

    ####################################################################################################################
    print("evaluating...")

    model.eval()
    
    images_dir = evaldir

    all_frames = []
    psnrs = []

    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        if eval_dataset.has_groundtruth:
            out_img_name = os.path.basename(eval_dataset.image_paths[indices[0]])[:-4]
        else:
            out_img_name = '{}'.format(indices[0])
        
        if len(kwargs['view_name']) > 0 and out_img_name != kwargs['view_name']:
            print('Skipping: ', out_img_name)
            continue

        print('Evaluating data_index: ', data_index, len(eval_dataloader))

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        model_input["envmap"] = model_input["envmap"].cuda()
        if relight == True:
            model_input["envmap"] = new_envmap.cuda()

        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            out = model(s)
            res.append({'points': out['points'].detach(),
                        'network_object_mask': out['network_object_mask'].detach(),
                        'object_mask': out['object_mask'].detach(),
                        'mask_sdf_output': out['mask_sdf_output'].detach(),
                        'normal_values': out['normal_values'].detach(),
                        'rgb_values': out['rgb_values'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        normal = model_outputs['normal_values']
        normal = normal.reshape(batch_size, total_pixels, 3)
        normal = (normal + 1.) / 2.
        normal_torch = plt.lin2img(normal, img_res)
        normal = normal_torch.detach().cpu().numpy()[0]
        normal = normal.transpose(1, 2, 0)
        import torchvision.utils as vutils
        if kwargs['save_exr']:
            imageio.imwrite('{0}/normal_{1}.exr'.format(images_dir, out_img_name), normal)

        else:
            img = Image.fromarray((normal * 255).astype(np.uint8))
            img.save('{0}/normal_{1}.png'.format(images_dir, out_img_name))
            

        gamma = 1.0

        tonemap_img = lambda x: np.power(x, 1./gamma)
        clip_img = lambda x: np.clip(x, 0., 1.)

        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        
        rgb_eval = clip_img(tonemap_img(rgb_eval))
        img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
        img.save('{0}/rgb_{1}.png'.format(images_dir, out_img_name))

        all_frames.append(np.array(img))

        rgb_gt = ground_truth['rgb']
        rgb_gt = plt.lin2img(rgb_gt, img_res).numpy()[0].transpose(1, 2, 0)
        
        rgb_gt = clip_img(tonemap_img(rgb_gt))
        img = Image.fromarray((rgb_gt * 255).astype(np.uint8))
        img.save('{0}/gt_{1}.png'.format(images_dir, out_img_name))

        mask = model_input['object_mask']
        mask = plt.lin2img(mask.unsqueeze(-1), img_res).cpu().numpy()[0]
        mask = mask.transpose(1, 2, 0)
        rgb_eval_masked = rgb_eval * mask
        rgb_gt_masked = rgb_gt * mask

        psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, mask)
        psnrs.append(psnr)

    if len(psnrs) > 0:
        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std()))



def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--view_name', type=str, default='', help='')
    parser.add_argument('--save_exr', default=False, action="store_true", help='')
    parser.add_argument('--exps_folder', type=str, default='pipeline_exps', help='The experiments folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--envmap_path', type=str, default='',  help='relight envmap')
    
    
    
    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
            data_split_dir=opt.data_split_dir,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             resolution=opt.resolution,
             envmap_path=opt.envmap_path,
             view_name=opt.view_name,
             save_exr=opt.save_exr,
             )
