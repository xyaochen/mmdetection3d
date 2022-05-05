import torch

from mmcv.utils import ext_loader


ext_module = ext_loader.load_ext('_ext', [
    'iou3d_boxes_iou_bev_forward', 'iou3d_nms_forward',
    'iou3d_nms_normal_forward'
])

def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None):
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.
    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Default: None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Default: None.
    Returns:
        torch.Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]

    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = torch.zeros(size=(), dtype=torch.long)
    ext_module.iou3d_nms_forward(
        boxes, keep, num_out, nms_overlap_thresh=thresh)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep