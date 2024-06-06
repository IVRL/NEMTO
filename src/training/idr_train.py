import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np

import utils.general as utils
import utils.plots as plt
from tensorboardX import SummaryWriter

class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
 
        self.expname = self.conf.get_string('train.expname') + '-' + kwargs['expname']
        self.freeze_geometry_epoch = 200

       
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']
        

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')
        
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['data_split_dir'])
 
        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        print("before setup model")
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))


        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        
        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])


        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):
        print("training...")

        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)
         
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch % 100 == 0:
                self.save_checkpoints(epoch)

            if epoch >= self.freeze_geometry_epoch:
                print('Now we are freezing geometry and optimizing refraction after epoch 200!!!')
                self.model.freeze_geometry()
                self.model.refraction_net.unfreeze_all()

                self.loss.rgb_weight = 100.
                self.loss.refraction_weight = 100.
                self.loss.refraction_smooth_weight = 10.

                self.loss.eikonal_weight = 0.
                self.loss.mask_weight = 0.
                self.loss.normalsmooth_weight = 0.


            if epoch < self.freeze_geometry_epoch:
                print("We are not yet optimizing for IOR")
                self.model.refraction_net.freeze_all()

                self.loss.rgb_weight = 0.
                self.loss.refraction_weight = 0.
                self.loss.refraction_smooth_weight = 0.
                self.loss.normalsmooth_weight = 0.


            if epoch % self.plot_freq == 0:
                self.model.eval()
                self.train_dataset.change_sampling_idx(-1)
                
                for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):

                    if not indices==8:
                        continue 

                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input["object_mask"] = model_input["object_mask"].cuda()
                    model_input['pose'] = model_input['pose'].cuda()
                    model_input["envmap"] = model_input["envmap"].cuda()
                    split = utils.split_input(model_input, self.total_pixels)
                    res = []
                    for s in split:
                        out = self.model(s)
                        res.append({
                            'points': out['points'].detach(),
                            'network_object_mask': out['network_object_mask'].detach(),
                            'object_mask': out['object_mask'].detach(),

                            'mask_sdf_output': out['mask_sdf_output'].detach(),
                            'normal_values': out['normal_values'].detach(),
                            'rgb_values': out['rgb_values'].detach(),
                        })

                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                    plt.plot(self.model,
                            indices,
                            model_outputs,
                            model_input['pose'],
                            ground_truth['rgb'],
                            self.plots_dir,
                            epoch,
                            self.img_res,
                            **self.plot_conf
                            )

                self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            loss_to_write = None

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input["envmap"] = model_input["envmap"].cuda()

                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                print(
                    '{0} [{1}] ({2}/{3}): loss = {4}, eikonal_loss = {5}, mask_loss = {6}, alpha = {7}, lr = {8}, rgb_loss = {9}, normalsmooth_loss = {10}, refraction_loss={11}, refraction_smooth_loss={12}'
                        .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['eikonal_loss'].item() * self.loss.eikonal_weight,
                                loss_output['mask_loss'].item() * self.loss.mask_weight,
                                self.loss.alpha,
                                self.scheduler.get_lr()[0],
                                loss_output['rgb_loss'].item() * self.loss.rgb_weight,
                                loss_output['normalsmooth_loss'].item() * self.loss.normalsmooth_weight, 
                                loss_output['refraction_loss'].item() * self.loss.refraction_weight, 
                                loss_output['refraction_smooth_loss'].item() * self.loss.refraction_smooth_weight, 
                                ))


                loss_to_write = loss_output
                
            self.writer.add_scalar('eikonal_loss', loss_to_write['eikonal_loss'].item(), epoch)
            self.writer.add_scalar('mask_loss', loss_to_write['mask_loss'].item(), epoch)
            self.writer.add_scalar('alpha', self.loss.alpha, epoch)
            self.writer.add_scalar('mask_weight', self.loss.mask_weight, epoch)
            self.writer.add_scalar('eikonal_weight', self.loss.eikonal_weight, epoch)
            self.writer.add_scalar('rgb_weight', self.loss.rgb_weight, epoch)
            self.writer.add_scalar('normalsmooth_weight', self.loss.normalsmooth_weight, epoch)
            self.writer.add_scalar('refraction_weight', self.loss.refraction_weight, epoch)
            self.writer.add_scalar('refraction_smooth_weight', self.loss.refraction_smooth_weight, epoch)


            self.writer.add_scalar('rgb_loss', loss_to_write['rgb_loss'], epoch)

            self.writer.add_scalar('lrate', self.scheduler.get_lr()[0], epoch)
            self.writer.add_scalar('ior', self.model.refraction_net.get_ior(), epoch)
            self.writer.add_scalar('refraction_loss', loss_to_write['refraction_loss'].item(), epoch)
            self.writer.add_scalar('normalsmooth_loss', loss_to_write['normalsmooth_loss'].item(), epoch)
            self.writer.add_scalar('refraction_smooth_loss', loss_to_write['refraction_smooth_loss'].item(), epoch)
            

            self.scheduler.step()
