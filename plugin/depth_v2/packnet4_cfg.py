from mmdet.models import DETECTORS
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models.backbones import ResNetV1d
from .utils import DepthPredictHead2Up, get_depth_metrics

@DETECTORS.register_module()
class ResDepthModel(nn.Module):

    def __init__(self, depth=50,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 2),
                out_indices=(0, 1, 2, 3),
                base_channels=64,
                **kwargs):
        super(ResDepthModel, self).__init__()
        self.backbone = ResNetV1d(depth=depth,
                            strides=strides,
                            dilations=dilations,
                            out_indices=out_indices,
                            base_channels=base_channels)
        # output channel =

        feat_dim = self.backbone.feat_dim
        self.head = DepthPredictHead2Up(feat_dim)


    def get_pred(self, data):
        x = data['img']

        features = self.backbone(x)

        last_feat = features[-1]

        pred = self.head(last_feat)

        return pred

    def forward(self, return_loss=True, rescale=False, **kwargs):

        if not return_loss:
            # in evalhook!

            x = kwargs['img']
            label = kwargs['depth_map']

            data = {'img':x, 'depth_map':label}
            depth_pred = self.get_pred(data)
            label = data['depth_map'].unsqueeze(dim=1)
            mask = (label > 0)

            #print(depth_pred.shape, label.shape, mask.shape, 'data shape')
            loss = torch.abs((label - depth_pred)) * mask
            loss = torch.sum(loss) / torch.sum(mask)

            with torch.no_grad():
                metrics = get_depth_metrics(depth_pred, label, mask)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]

            # hack the hook
            # outputs[0]=None. see https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/test.py#L99
            #outputs = {'loss': loss, 'log_vars':log_vars, 'num_samples':depth_pred.size(0), 0:None}
            #print('val', loss)
            metrics.append(loss.item())
            return [metrics]
        raise NotImplementedError

    def train_step(self, data, optimzier):
        depth_pred = self.get_pred(data)
        label = data['depth_map'].unsqueeze(dim=1)
        mask = (label > 0)

        #print(depth_pred.shape, label.shape, mask.shape, 'data shape')
        #from IPython import embed
        #embed()
        loss = torch.abs((label - depth_pred)) * mask

        loss = torch.sum(loss) / torch.sum(mask)

        log_var = {}
        with torch.no_grad():
                metrics = get_depth_metrics(depth_pred, label, mask)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]
                abs_diff, abs_rel, sq_rel, rmse, rmse_log = metrics
        sparsity = torch.sum(mask) * 1.0 / torch.numel(mask)


        std = torch.tensor([58.395, 57.12, 57.375]).cuda().view(1, -1, 1, 1)
        mean = torch.tensor([123.675, 116.28, 103.53]).cuda().view(1, -1, 1, 1)
        img = data['img'] * std + mean
        img = img / 255.0
        depth_at_gt = depth_pred * mask
        log_vars = {'loss': loss.item(), 'sparsity': sparsity.item(),
                    'abs_diff': abs_diff, 'abs_rel': abs_rel,
                    'sq_rel': sq_rel, 'rmse': rmse,
                    'rmse_log': rmse_log
                     }
        # 'pred', 'data', 'label', 'depth_at_gt' is used for visualization only!
        outputs = {'pred': torch.clamp(1.0 / (depth_pred+1e-4), 0, 1), 'data': img,
                'label': torch.clamp(1.0 / (label+1e-4), 0, 1),
                'depth_at_gt': torch.clamp(1.0 / (depth_at_gt+1e-4), 0., 1),
                'loss':loss, 'log_vars':log_vars, 'num_samples':depth_pred.size(0)}

        return outputs

    def val_step(self, data, optimizer):

        return self.train_step(self, data, optimizer)
