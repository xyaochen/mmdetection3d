# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from plugin.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.cnn import ConvModule, build_conv_layer

def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    reference_points = reference_points.clone()
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    B, num_query = reference_points.size()[:2]
    
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    
    num_cam = lidar2img.size(1)
    # ref_point change to (B, num_cam, num_query, 4, 1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # lidar2img chaneg to (B, num_cam, num_query, 4, 4)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    # ref_point_cam change to (B, num_cam, num_query, 4)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    # ref_point_cam change to img coordinates
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    # img_metas['img_shape']=[900, 1600]
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    # mask shape (B, 1, num_query, num_cam, 1, 1)
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    num_points = 1
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        # ref_point_cam shape change from (B, num_cam, num_query, 2) to (B*num_cam, num_query/10, 10, 2)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2).to(feat.device)
        # sample_feat shape (B*N, C, num_query/10, 10)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # sampled_feat shape (B, C, num_query, N, num_points)
        sampled_feat = sampled_feat.view(B, N, C, num_query, num_points).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, num_points, len(mlvl_feats))
    # ref_point_3d (B, N, num_query, 3)  maks (B, N, num_query, 1)
    return sampled_feats

@DETECTORS.register_module()
class CenterPointBEVAug(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 freeze_encoder=False,
                 embed_dims=256,
                 num_cams=6,
                 num_levels=4,
                 z_size=4,
                 BEV_size_factor=4,
                 flip=False,
                 rotation=False,
                 scale=False,
                 init_cfg=None,
                 ):
        super(CenterPointBEVAug,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.freeze_encoder = freeze_encoder
        self.flip = flip
        self.rotation = rotation
        self.scale = scale
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_levels = num_levels

        if self.flip:
            self.flip_horizontal = True if np.random.rand(
                ) < 0.5 else False
            self.flip_vertical = True if np.random.rand(
                ) < 0.5 else False

        self.pc_range = self.train_cfg['pts']['point_cloud_range']
        self.grid_size = self.train_cfg['pts']['grid_size']
        self.out_size_factor = self.train_cfg['pts']['out_size_factor']
        self.BEV_size_factor = BEV_size_factor
        self.x_size = self.grid_size[0] // self.BEV_size_factor
        self.y_size = self.grid_size[1] // self.BEV_size_factor
        self.z_size = z_size

        self.BEV_query = nn.Embedding(self.x_size*self.y_size, embed_dims)
        #self.offsets = nn.Linear(embed_dims, num_points*3)
        self.position_enc = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.attention_weights = nn.Linear(embed_dims, num_cams*num_levels)

        self.BEV_enc = nn.Sequential(
            ConvModule(
                embed_dims,
                embed_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias='auto',
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d')),
            ConvModule(
                embed_dims,
                embed_dims,
                kernel_size=3,
                stride=2,
                padding=1,
                bias='auto',
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d')),
        )

    def init_weights(self):
        """Initialize model weights."""
        if self.freeze_encoder:
            #if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
                
            #if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            #img.requires_grad = True
            if self.freeze_encoder:
                with torch.no_grad():    
                    img_feats = self.img_backbone(img)
            else:
                img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            if self.freeze_encoder:
                with torch.no_grad():  
                    img_feats = self.img_neck(img_feats)
            else:
                img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_img_feat(img, img_metas)
        losses = dict()
        
        BEV_feats = self.creat_BEV(img_feats, img_metas)

        #print(type(gt_bboxes_3d), len(gt_bboxes_3d))
        #print(type(gt_bboxes_3d[0]), len(gt_bboxes_3d[0]))

        if self.flip:
            if self.flip_horizontal:
                BEV_feats = [torch.flip(BEV_feat, dim=3) for BEV_feat in BEV_feats]
                for gt_bboxes in gt_bboxes_3d:
                    for gt_bbox in gt_bboxes:
                        gt_bbox.flip('horizontal')
            if self.flip_vertical:
                BEV_feats = [torch.flip(BEV_feat, dim=2) for BEV_feat in BEV_feats]
                for gt_bboxes in gt_bboxes_3d:
                    for gt_bbox in gt_bboxes:
                        gt_bbox.flip('vertical')
        
        BEV_feats = [self.BEV_enc(BEV_feat) for BEV_feat in BEV_feats]

        losses_pts = self.forward_pts_train(BEV_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses

    def creat_BEV(self, img_feats, img_metas):

        batch_size = img_feats[0].size(0)

        BEV_query = self.BEV_query.weight
        BEV_query = BEV_query.view(1, self.x_size, self.y_size, self.embed_dims).expand(batch_size, -1, -1, -1)
        # (B, x_size*y_size*z_size, 3)
        reference_points = self.create_3D_grid(self.x_size, self.y_size, 4).expand(batch_size, -1, -1)
        reference_points[..., 0] /= self.x_size
        reference_points[..., 1] /= self.y_size
        reference_points[..., 2] /= 4

        reference_points = reference_points.to(img_feats[0].device)

        #offsets = self.offsets(BEV_query)
        PE_embed = self.position_enc(reference_points).view(batch_size, self.embed_dims, self.y_size, self.x_size, self.z_size)
        PE_embed = PE_embed.sum(-1)

        num_query = self.x_size * self.y_size
        attention_weights = self.attention_weights(BEV_query).view(batch_size, 1, num_query, 1, self.num_cams, 1, self.num_levels)
        
        # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
        BEV_feats = feature_sampling(img_feats, reference_points, self.pc_range, img_metas)
        BEV_feats = BEV_feats.view(batch_size, self.embed_dims, num_query, self.z_size, self.num_cams, 1, self.num_levels)
        BEV_feats = BEV_feats * attention_weights

        BEV_feats = BEV_feats.sum(-1).sum(-1).sum(-1).sum(-1)
        BEV_feats = BEV_feats.view(batch_size, self.embed_dims, self.x_size, self.y_size)
        BEV_feats = BEV_feats.permute(0, 1, 3, 2)
        
        BEV_query = BEV_query.permute(0, 3, 2, 1)
        BEV_feats = BEV_query + BEV_feats + PE_embed
        BEV_feats = [BEV_feats]

        return BEV_feats

    def create_3D_grid(self, x_size, y_size, z_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size], [0, z_size - 1, z_size]]
        batch_x, batch_y, batch_z = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None], batch_z[None]], dim=0)[None]
        coord_base = coord_base.view(1, 3, -1).permute(0, 2, 1)
        return coord_base

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_img_feat(
            img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return bbox_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.
        The function implementation process is as follows:
            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.
        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool, optional): Whether to rescale bboxes.
                Default: False.
        Returns:
            dict: Returned bboxes consists of the following keys:
                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]