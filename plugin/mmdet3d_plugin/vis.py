import numpy as np
import torch
import time
from torchvision import utils as vutils

def save_bev(pts , ref, data_root):
    if isinstance(pts, list):
        pts = pts[0]
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    mask = ((pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) & 
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] > pc_range[2]) & (pts[:, 2] < pc_range[5]))
    pts = pts[mask]

    res = 0.1
    x_max = 1 + int((pc_range[3] - pc_range[0]) / res)
    y_max = 1 + int((pc_range[4] - pc_range[1]) / res)
    im = torch.zeros(x_max+1, y_max+1, 3)

    x_img = (pts[:, 0] - pc_range[0]) / res
    x_img = x_img.round().long()
    y_img = (pts[:, 1] - pc_range[1]) / res
    y_img = y_img.round().long()

    im[x_img, y_img, :] = 1
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            im[(x_img.long()+i).clamp(min=0, max=x_max), 
                (y_img.long()+j).clamp(min=0, max=y_max), :] = 1
    print('reference', ref.size())
    ref_pts_x = (ref[..., 0] * (pc_range[3] - pc_range[0]) / res).round().long()
    ref_pts_y = (ref[..., 1] * (pc_range[4] - pc_range[1]) / res).round().long()

    for i in [-2, 0, 2]:
        for j in [-2, 0, 2]:
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 0] = 1
            im[(ref_pts_x.long()+i).clamp(min=0, max=x_max), 
                (ref_pts_y.long()+j).clamp(min=0, max=y_max), 1:2] = 0

    im = im.permute(2, 0, 1)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    print(timestamp)
    # saved_root = '/home/chenxy/mmdetection3d/'
    vutils.save_image(im, data_root + '/' + timestamp + '.jpg')