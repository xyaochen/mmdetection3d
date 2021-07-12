import torch
import numpy as np

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import Hook, HOOKS
from mmdet.datasets.builder import PIPELINES
from mmcv.runner.hooks.logger.base import LoggerHook

@HOOKS.register_module()
class ProgressiveScaling(Hook):
    """
    Helper class to manage progressive scaling.
    After a certain training progress percentage, decrease the number of scales by 1.
    Parameters
    ----------
    progressive_scaling : float
        Training progress percentage where the number of scales is decreased
    num_scales : int
        Initial number of scales
    """
    def __init__(self, progressive_scaling=0.2, num_scales=4):
        #super(ProgressiveScaling, self).__init__()
        self.num_scales = 4
        self.progressive_scaling = progressive_scaling
        # Use it only if bigger than zero (make a list)
        if progressive_scaling > 0.0:
            self.progressive_scaling = np.float32(
                [progressive_scaling * (i + 1) for i in range(num_scales - 1)] + [1.0])
        # Otherwise, disable it
        else:
            self.progressive_scaling = progressive_scaling
    def get_num_scales(self, runner):
        """
        Call for an update in the number of scales
        Parameters
        ----------
        progress : float
            Training progress percentage
        Returns
        -------
        num_scales : int
            New number of scales
        """
        #print(type(runner), runner)
        progress = runner.epoch / runner._max_epochs
        print('epoch', runner.epoch, progress)
        if isinstance(self.progressive_scaling, list):
            return int(self.num_scales -
                       np.searchsorted(self.progressive_scaling, progress))
        else:
            return self.num_scales

    def before_epoch(self, runner):
        print(type(runner.model), runner.model.num_scales)
        runner.model.num_scales = self.get_num_scales(runner)
        
