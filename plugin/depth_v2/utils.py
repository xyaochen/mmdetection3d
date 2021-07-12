import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(out_planes)
    )
class DepthPredictHead2Up(nn.Module):
    '''
    1. We use a softplus activation to generate positive depths.
        The predicted depth is no longer bounded.
    2. The network predicts depth rather than disparity, and at a single scale.
    '''

    def __init__(self, in_channels):
        super(DepthPredictHead2Up, self).__init__()

        self.up = nn.PixelShuffle(2)
        self.conv1 = conv(in_channels//4, in_channels//4, kernel_size=3)
        self.conv2 = conv(in_channels//16, in_channels//16, kernel_size=3)
        self.conv3 = conv(in_channels//64, in_channels//64, kernel_size=3)
        self.conv4 = conv(in_channels//64, in_channels//64, kernel_size=3)
        self.conv5 = conv(in_channels//64, 1, kernel_size=1, padding=0)


    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

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

def inv_val(img):
    if isinstance(img, list):
        inv_img = []
        for v in img:
            inv_img.append(1. / v.clamp(min=1e-6))
    else :
        inv_img = 1. / img.clamp(min=1e-6)
    return inv_img

def remap_invdepth_color(disp):
    '''
    disp: torch.Tensor [1, H, W]
    '''

    disp_np = disp.squeeze().cpu().numpy()
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')

    # colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # shape [H, W, 3]
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3]) 

    return colormapped_im

def match_scales(image, targets, num_scales,
                 mode='bilinear', align_corners=True):
    """
    Interpolate one image to produce a list of images with the same shape as targets
    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Input image
    targets : list of torch.Tensor [B,?,?,?]
        Tensors with the target resolutions
    num_scales : int
        Number of considered scales
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation
    Returns
    -------
    images : list of torch.Tensor [B,?,?,?]
        List of images with the same resolutions as targets
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape[-2:]
        # If image shape is equal to target shape
        if image_shape == target_shape:
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(nn.functional.interpolate(
                image, size=target_shape, mode=mode, align_corners=align_corners))
    # Return scaled images
    return images

def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def inv_depths_normalize(inv_depths):
    """
    Inverse depth normalization
    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps
    Returns
    -------
    norm_inv_depths : list of torch.Tensor [B,1,H,W]
        Normalized inverse depth maps
    """
    mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
    return [inv_depth / mean_inv_depth.clamp(min=1e-6)
            for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)]


def calc_smoothness(inv_depths, images, num_scales):
    """
    Calculate smoothness values for inverse depths
    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps
    images : list of torch.Tensor [B,3,H,W]
        Inverse depth maps
    num_scales : int
        Number of scales considered
    Returns
    -------
    smoothness_x : list of torch.Tensor [B,1,H,W]
        Smoothness values in direction x
    smoothness_y : list of torch.Tensor [B,1,H,W]
        Smoothness values in direction y
    """
    inv_depths_norm = inv_depths_normalize(inv_depths)
    inv_depth_gradients_x = [gradient_x(d) for d in inv_depths_norm]
    inv_depth_gradients_y = [gradient_y(d) for d in inv_depths_norm]

    image_gradients_x = [gradient_x(image) for image in images]
    image_gradients_y = [gradient_y(image) for image in images]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    # Note: Fix gradient addition
    smoothness_x = [inv_depth_gradients_x[i] * weights_x[i] for i in range(num_scales)]
    smoothness_y = [inv_depth_gradients_y[i] * weights_y[i] for i in range(num_scales)]
    return smoothness_x, smoothness_y

def get_smooth_loss(inv_preds, img, num_scales=4):
    """
    Calculates the smoothness loss for inverse depth maps.
    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Predicted inverse depth maps for all scales
    images : list of torch.Tensor [B,3,H,W]
        Original images for all scales
    Returns
    -------
    smoothness_loss : torch.Tensor [1]
        Smoothness loss
    """
    images = []
    images = match_scales(img, inv_preds, num_scales, mode='bilinear', align_corners=True)
    # Calculate smoothness gradients
    smoothness_x, smoothness_y = calc_smoothness(inv_preds, images, num_scales)
    # Calculate smoothness loss
    smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                           smoothness_y[i].abs().mean()) / 2 ** i
                           for i in range(num_scales)]) / num_scales
    return smoothness_loss

def get_smooth_L2_loss(preds, img):
    loss = 0
    B, _, H, W = img.shape
    img_dx = gradient_x(img)
    img_dy = gradient_y(img)
    weights_x = torch.exp(-torch.mean(abs(img_dx), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(abs(img_dy), 1, keepdim=True))
    weights = 1
    decay = 1.0 / 4.0
    if isinstance(preds, list):
        for pred in preds:
            up_pred = nn.functional.interpolate(pred, size=[H, W])
            dep_dx = abs(gradient_x(up_pred))
            dep_dy = abs(gradient_y(up_pred))
            loss1 = weights * torch.sum((dep_dx ** 2) * weights_x) / torch.numel(dep_dx)
            loss1 += weights * torch.sum((dep_dy ** 2) * weights_y) / torch.numel(dep_dy)
            loss += loss1
            weights = weights * decay
    else:
        dep_dx = abs(gradient_x(preds))
        dep_dy = abs(gradient_y(preds))
        loss = weights * torch.sum((dep_dx ** 2) * weights_x) / torch.numel(dep_dx)
        loss += weights * torch.sum((dep_dy ** 2) * weights_y) / torch.numel(dep_dy)
    return loss
'''
def get_L1_loss(preds, label, mask):
    loss = 0
    num = torch.sum(mask)
    num = num * 1.0
    #print(label.shape)
    B, _, H, W = label.shape
    if isinstance(preds, list):     #train
        for pred in preds:
            #print(pred.shape)
            up_pred = nn.functional.interpolate(pred, size=[H, W], mode='bilinear')
            loss1 = torch.abs((label - up_pred)) * mask
            loss1 = torch.sum(loss1) / num
            loss += loss1
    else:                           #test
        loss = torch.abs((label - preds)) * mask
        loss = torch.sum(loss) / num
    return loss
'''

def get_L1_loss(inv_preds, gt_inv_depth, num_scales):
    loss = 0
    gt_inv_depths = match_scales(gt_inv_depth, inv_preds, num_scales, mode='nearest', align_corners=None)
    for i in range(num_scales):
        mask = (gt_inv_depths[i] < 1e6)
        num = mask.sum()
        #loss += (abs(inv_preds[i] - gt_inv_depths[i]) * mask).sum() / (num * (2 ** i))
        loss += (abs(inv_preds[i] - gt_inv_depths[i]) * mask).sum() / num
    loss /= num_scales
    return loss

