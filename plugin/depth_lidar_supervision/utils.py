import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.
    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.
    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out

class DepthPredictHead(nn.Module):
    '''
    1. We use a softplus activation to generate positive depths. 
        The predicted depth is no longer bounded.
    2. The network predicts depth rather than disparity, and at a single scale.
    '''

    def __init__(self, in_channels):
        super(DepthPredictHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    
    def forward(self, x):
        
        x = self.conv(x)

        pred = nn.functional.softplus(x)

        return pred


def get_depth_metrics(pred, gt, mask=None):
    """
    params:
    pred: [N,1,H,W].  torch.Tensor
    gt: [N,1,H,W].     torch.Tensor
    """
    if mask is not None:
        num = torch.sum(mask) # the number of non-zeros
        pred = pred[mask]
        gt = gt[mask]
    else:
        num = pred.numel()

    num = num * 1.0
    diff_i = gt - pred

    abs_diff = torch.sum(torch.abs(diff_i)) / num
    abs_rel = torch.sum(torch.abs(diff_i) / gt) / num
    sq_rel = torch.sum(diff_i ** 2 / gt) / num
    rmse = torch.sqrt(torch.sum(diff_i ** 2) / num)
    rmse_log = torch.sqrt(torch.sum((torch.log(gt) -
                                        torch.log(pred)) ** 2) / num)
    
    return abs_diff, abs_rel, sq_rel, rmse, rmse_log