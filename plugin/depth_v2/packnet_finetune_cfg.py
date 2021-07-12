# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from .layers01 import \
    PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth
from .utils import get_smooth_L2_loss, get_smooth_loss
from mmdet.models import DETECTORS
from .utils import get_L1_loss, get_smooth_loss, get_smooth_L2_loss, \
    remap_invdepth_color, get_depth_metrics

@DETECTORS.register_module()
class PackNetSlim04(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).
    Slimmer version, with fewer feature channels

    https://arxiv.org/abs/1905.02693

    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, min_depth=0.5, **kwargs):
        super().__init__()
        self.version = version
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, no = 32, out_channels
        n1, n2, n3, n4, n5 = 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)
        # Support for different versions
        if self.version == 'A':  # Channel concatenation
            n1o, n1i = n1, n1 + ni + no
            n2o, n2i = n2, n2 + n1 + no
            n3o, n3i = n3, n3 + n2 + no
            n4o, n4i = n4, n4 + n3
            n5o, n5i = n5, n5 + n4
        elif self.version == 'B':  # Channel addition
            n1o, n1i = n1, n1 + no
            n2o, n2i = n2, n2 + no
            n3o, n3i = n3//2, n3//2 + no
            n4o, n4i = n4//2, n4//2
            n5o, n5i = n5//2, n5//2
        else:
            raise ValueError('Unknown PackNet version {}'.format(version))

        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

        # Decoder

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels, min_depth=min_depth)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels, min_depth=min_depth)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels, min_depth=min_depth)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels, min_depth=min_depth)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_pred(self, rgb, **kwargs):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        inv_depth4 = self.disp4_layer(iconv4)
        up_inv_depth4 = self.unpack_disp4(inv_depth4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, up_inv_depth4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, up_inv_depth4), 1)
        iconv3 = self.iconv3(concat3)
        inv_depth3 = self.disp3_layer(iconv3)
        up_inv_depth3 = self.unpack_disp3(inv_depth3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, up_inv_depth3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, up_inv_depth3), 1)
        iconv2 = self.iconv2(concat2)
        inv_depth2 = self.disp2_layer(iconv2)
        up_inv_depth2 = self.unpack_disp2(inv_depth2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, up_inv_depth2), 1)
        else:
            concat1 = torch.cat((unpack1 + skip1, up_inv_depth2), 1)
        iconv1 = self.iconv1(concat1)
        inv_depth1 = self.disp1_layer(iconv1)

        inv_depths = [inv_depth1, inv_depth2, inv_depth3, inv_depth4]

        return inv_depths

    #计算值的inverse
    def inv_val(self, x):
        y = []
        for v in x:
            y.append(1. / v.clamp(min=1e-6))   #避免除数为0
        return y

    def forward(self, return_loss=True, rescale=False, **kwargs):
        
        if not return_loss:
            # in evalhook!

            x = kwargs['img']
            label = kwargs['depth_map']

            data = {'img':x, 'depth_map':label}
            inv_depth_pred = self.get_pred(data['img'])
            depth_pred = self.inv_val(inv_depth_pred)
            label = data['depth_map'].unsqueeze(dim=1)
            mask = (label > 0)

            #smoothness_loss = get_smooth_loss(depth_pred, data['img'])
            smoothness_loss = get_smooth_loss(depth_pred[0], data['img'])
            L1_loss = get_L1_loss(depth_pred[0], label, mask)
            loss = L1_loss + smoothness_loss
            #print('L1_loss', depth_pred[0].shape, label.shape, mask.shape)
            with torch.no_grad():
                metrics = get_depth_metrics(depth_pred[0], label, mask)
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
        inv_depth_pred = self.get_pred(data['img'])
        depth_pred = self.inv_val(inv_depth_pred)
        label = data['depth_map'].unsqueeze(dim=1)
        mask = (label > 0)

        smoothness_loss = get_smooth_loss(depth_pred[0], data['img'])
        L1_loss = get_L1_loss(depth_pred[0], label, mask)
        loss = L1_loss + smoothness_loss
        B, _, H, W = label.shape
        #print('train_loss', depth_pred[0].shape)
        with torch.no_grad():
                metrics = get_depth_metrics(depth_pred[0], label, mask)
                # abs_diff, abs_rel, sq_rel, rmse, rmse_log
                metrics = [m.item() for m in metrics]
                abs_diff, abs_rel, sq_rel, rmse, rmse_log = metrics

                out_img1 = remap_invdepth_color(inv_depth_pred[0][0].clamp(max=1.0))
                out_img2 = remap_invdepth_color(inv_depth_pred[1][0].clamp(max=1.0))
                out_img2_up = remap_invdepth_color(nn.functional.interpolate(inv_depth_pred[1].clamp(max=1.0), size=[H, W])[0])
                out_img3 = remap_invdepth_color(inv_depth_pred[2][0].clamp(max=1.0))
                out_img3_up = remap_invdepth_color(nn.functional.interpolate(inv_depth_pred[2].clamp(max=1.0), size=[H, W])[0])
                out_img4 = remap_invdepth_color(inv_depth_pred[3][0].clamp(max=1.0))
                out_img4_up = remap_invdepth_color(nn.functional.interpolate(inv_depth_pred[3].clamp(max=1.0), size=[H, W])[0])
                #print(out_img.shape)  (448, 768, 3)
        
        sparsity = torch.sum(mask) * 1.0 / torch.numel(mask)

        std = torch.tensor([58.395, 57.12, 57.375]).cuda().view(1, -1, 1, 1)
        mean = torch.tensor([123.675, 116.28, 103.53]).cuda().view(1, -1, 1, 1)
        img = data['img'] * std + mean
        img = img / 255.0
        depth_at_gt = depth_pred[0] * mask

        log_vars = {'loss': loss.item(), 'L1_loss': L1_loss.item(), 
                    'sm_loss': smoothness_loss.item(), 'sparsity': sparsity.item(),
                    'abs_diff': abs_diff, 'abs_rel': abs_rel,
                    'sq_rel': sq_rel, 'rmse': rmse,
                    'rmse_log': rmse_log
                     }
        
        # 'pred', 'data', 'label', 'depth_at_gt' is used for visualization only!
        #print(img[0].shape, label[0].shape, depth_at_gt[0].shape)
        outputs = {'pred': out_img1.transpose(2, 0, 1), 'data': img[0],
                'out_img2': out_img2.transpose(2, 0, 1), 'out_img2_up': out_img2_up.transpose(2, 0, 1),
                'out_img3': out_img3.transpose(2, 0, 1), 'out_img3_up': out_img3_up.transpose(2, 0, 1),
                'out_img4': out_img4.transpose(2, 0, 1), 'out_img4_up': out_img4_up.transpose(2, 0, 1),
                'label': torch.clamp(1.0 / (label[0]+1e-4), 0, 1),
                'depth_at_gt': torch.clamp(1.0 / (depth_at_gt[0]+1e-4), 0., 1),
                'loss':loss, 'log_vars':log_vars, 'num_samples':depth_pred[0].size(0)}

        return outputs