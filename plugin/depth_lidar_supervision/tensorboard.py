import os.path as osp
import time
import torch
from torchvision import utils as vutils

from mmcv.utils import TORCH_VERSION
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module()
class TensorboardLoggerHook2(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(TensorboardLoggerHook2, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs' + timestamp)
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = f'{var}/{runner.mode}'
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)
        '''
        for var in runner.outputs:
            if var in ['pred', 'data']:
                tag = f'{var}/{runner.mode}'
                img = runner.outputs[var]
                self.writer.add_images(tag, img, runner.iter)
                if var == 'pred':
                    #img = torch.squeeze(img)
                    #for i in range(img.shape[0]):
                    vutils.save_image(img, var + '_img' + '.png')
                else:
                    #for i in range(img.shape[0]):
                    vutils.save_image(img, var + '_img' + '.png')
        '''
        # add learning rate
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                self.writer.add_scalar(f'learning_rate/{name}', value[0],
                                       runner.iter)
        else:
            self.writer.add_scalar('learning_rate', lrs[0], runner.iter)
        # add momentum
        momentums = runner.current_momentum()
        if isinstance(momentums, dict):
            for name, value in momentums.items():
                self.writer.add_scalar(f'momentum/{name}', value[0],
                                       runner.iter)
        else:
            self.writer.add_scalar('momentum', momentums[0], runner.iter)

    @master_only
    def after_epoch(self, runner):
        for var in runner.outputs:
            if var in ['pred', 'data']:
                tag = f'{var}/{runner.mode}'
                img = runner.outputs[var]
                self.writer.add_images(tag, img, runner.iter)
            if var == 'data':
                vutils.save_image(img, var + '_img' + str(runner.iter) + '.png')

    @master_only
    def after_run(self, runner):
        self.writer.close()
