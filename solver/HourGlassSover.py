import argparse, pprint, os
import numpy as np
import pandas as pd

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from utils import utils
from model import StackedHourGlass


class HourGlassSover(object):
    def __init__(self, opt):
        super(HourGlassSover, self).__init__()
        self.opt = opt
        self.is_train = opt['is_train']
        self.use_gpu = opt['use_gpu'] and torch.cuda.is_available()
        self.exp_root = opt['path']['exp_root']
        self.checkpoint_dir = opt['path']['checkpoint_dir']
        self.visual_dir = opt['path']['visual_dir']
        self.records = {'epoch': [],
                        'train_loss': [],
                        'val_loss': [],
                        'lr': [],
                        'nme': []
                       }
        self.best_epoch = 0
        self.cur_epoch = 1
        self.best_pred = 1.0
        self.model = self._create_model(opt['networks'])
        if self.use_gpu:
            self.model = net = nn.DataParallel(self.model).cuda()

        self.print_network()
        if self.is_train:
            self.train_opt = opt['train']
            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion = nn.MSELoss()
            else:
                raise NotImplementedError(
                    'Loss type [%s] is not implemented!' % loss_type)
            if self.use_gpu:
                self.criterion = self.criterion.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt[
                'weight_decay'] else 0
            optim_type = self.train_opt['optimizer'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.train_opt['learning_rate'],
                    weight_decay=weight_decay)
            elif optim_type == "RMSPROP":
                self.optimizer = optim.RMSprop(
                    self.model.parameters(),
                    lr=self.train_opt['learning_rate'],
                    weight_decay=weight_decay)
            else:
                raise NotImplementedError(
                    'Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, self.train_opt['lr_steps'],
                    self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError(
                    'Only MultiStepLR scheme is supported!')

        self.load()
        
        print('===> Solver Initialized : [%s] || Use GPU : [%s]' %
              (self.__class__.__name__, self.use_gpu))

        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f" %
                  (self.scheduler.milestones, self.scheduler.gamma))

    def feed_data(self, batch):
        self.sample = batch['img']
        self.target = batch['heatmap_gt']
        self.path = batch['path']
        self.landmark_gt = batch['landmark_gt']
        if self.use_gpu:
            self.sample = self.sample.float().cuda()
            self.target = self.target.float().cuda()

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        output_list = self.model(self.sample)
        loss = 0.0
        for output in output_list:
            loss += self.criterion(output, self.target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            output_list = self.model(self.sample)
            loss = self.criterion(output_list[-1], self.target)
        landmarks = [utils.get_peak_points(x.cpu().numpy()) * 4 for x in output_list]
        self.landmarks = landmarks
        self.heatmaps = output_list
        return loss.item()

    def calc_nme(self):
        '''
        calculate normalized mean error
        '''
        landmark = self.landmarks[-1]
        diff = landmark - self.landmark_gt.numpy()
        nme = np.mean(np.sqrt(np.sum(np.square(diff), axis=2))) / self.sample.shape[-1]
        return nme

    def _create_model(self, opt):
        return StackedHourGlass(opt['hourglass'])

    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]' % filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' %
                  filename.replace('last_ckp', 'best_ckp'))
            torch.save(ckp, filename.replace('last_ckp', 'best_ckp'))

        if epoch % self.train_opt['save_interval'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' %
                  (epoch,
                   filename.replace('last_ckp', 'epoch_%d_ckp' % epoch)))

            torch.save(
                ckp, filename.replace('last_ckp', 'epoch_%d_ckp' % epoch))

    def load(self):
        """
        load or initialize network
        """
        if (self.is_train
                and self.opt['train']['pretrain']) or not self.is_train:
            model_path = self.opt['train']['pretrained_path']
            if model_path is None:
                raise ValueError(
                    "[Error] The 'pretrained_path' does not declarate in *.json"
                )

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['train']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.records = checkpoint['records']
                    self.best_pred = checkpoint['best_pred']
                    self.best_epoch = checkpoint['best_epoch']

            else:
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                load_func = self.model.module.load_state_dict if isinstance(self.model, nn.DataParallel) \
                    else self.model.load_state_dict
                load_func(checkpoint)

        # else:
            # self._net_init()

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

#     def get_current_log(self):
#         log = OrderedDict()
#         log['epoch'] = self.cur_epoch
#         log['best_pred'] = self.best_pred
#         log['best_epoch'] = self.best_epoch
#         log['records'] = self.records
#         return log

#     def set_current_log(self, log):
#         self.cur_epoch = log['epoch']
#         self.best_pred = log['best_pred']
#         self.best_epoch = log['best_epoch']
#         self.records = log['records']

    def save_current_log(self):
        data_frame = pd.DataFrame(
            data={'train_loss': self.records['train_loss']
                , 'val_loss': self.records['val_loss']
                , 'nme': self.records['nme']
                , 'lr': self.records['lr']
                  },
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.exp_root, 'train_records.csv'),
                          index_label='epoch')
        
    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.model.__class__.__name__,
                self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(
            net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'),
                      'w') as f:
                f.writelines(net_lines)

        print("==================================================")

    def get_current_visual(self):
        res_heatmaps = [
            np.squeeze(x.cpu().numpy(), axis=0) for x in self.heatmaps
        ]
        heatmap_gt = np.squeeze(self.target.cpu().numpy(), axis=0)
        img = np.squeeze(self.sample.cpu().numpy(), axis=0)
        mean = np.reshape(np.array(self.opt['datasets']['train']['mean']), (3, 1, 1))
        fig = utils.plot_heatmap_compare(res_heatmaps, heatmap_gt, img, mean)
        return fig
    
    def log_current_visual(self, img_name, tb_logger, current_step):
        res_heatmaps = [
            np.squeeze(x.cpu().numpy(), axis=0) for x in self.heatmaps
        ]
        heatmap_gt = np.squeeze(self.target.cpu().numpy(), axis=0)
        img = np.squeeze(self.sample.cpu().numpy(), axis=0)
        mean = np.reshape(np.array(self.opt['datasets']['train']['mean']), (3, 1, 1))
        fig = utils.plot_heatmap_compare(res_heatmaps, heatmap_gt, img, mean)
        tb_logger.add_figure(img_name, fig, global_step=current_step)

    def save_current_visual(self, img_name, epoch):
        save_dir = os.path.join(self.visual_dir, img_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        res_heatmaps = [
            np.squeeze(x.cpu().numpy(), axis=0) for x in self.heatmaps
        ]
        heatmap_gt = np.squeeze(self.target.cpu().numpy(), axis=0)
        img = np.squeeze(self.sample.cpu().numpy(), axis=0)
        mean = np.reshape(np.array(self.opt['datasets']['train']['mean']), (3, 1, 1))
        fig = utils.plot_heatmap_compare(res_heatmaps, heatmap_gt, img, mean)
        fig.savefig(os.path.join(save_dir, '%05d.png' % epoch))

    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
