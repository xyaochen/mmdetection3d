
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init, ConvModule, build_conv_layer
from mmcv.runner import force_fp32
                        
from mmdet.core import (multi_apply, build_assigner, build_sampler, 
                        multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DeformableDETRHead, DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.builder import HEADS, build_loss
from plugin.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox


@HEADS.register_module()
class FUTR3DInit2DHeadLiDAR(DETRHead):
    """Head of DeformDETR3DCam. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 *args,
                 nms_kernel_size,
                 initialize_by_heatmap=True,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 per_scene_noise=False,
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size

        super(FUTR3DInit2DHeadLiDAR, self).__init__(
            *args, transformer=transformer, **kwargs)

        self.loss_heatmap = build_loss(loss_heatmap)

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                self.embed_dims,
                self.embed_dims,
                kernel_size=3,
                padding=1,
                bias='auto',
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                self.embed_dims,
                self.num_classes,
                kernel_size=3,
                padding=1,
                bias='auto',
            ))
            self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(self.num_classes, self.embed_dims, 1)

        self.grid_size = self.train_cfg['grid_size']
        self.out_size_factor = self.train_cfg['out_size_factor']
        self.x_size = self.grid_size[0] // self.out_size_factor
        self.y_size = self.grid_size[1] // self.out_size_factor
        self.bev_pos = self.create_2D_grid(self.x_size, self.y_size)

        self.input_porj = build_conv_layer(
            dict(type='Conv2d'),
            self.in_channels,
            self.embed_dims,
            kernel_size=3,
            padding=1,
            bias='auto',
        )
        
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.per_scene_noise = per_scene_noise

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        center_reg_branch = []
        for _ in range(self.num_reg_fcs):
            center_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            center_reg_branch.append(nn.ReLU())
        center_reg_branch.append(Linear(self.embed_dims, 2))
        center_reg_branch = nn.Sequential(*center_reg_branch)

        ht_reg_branch = []
        for _ in range(self.num_reg_fcs):
            ht_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            ht_reg_branch.append(nn.ReLU())
        ht_reg_branch.append(Linear(self.embed_dims, 1))
        ht_reg_branch = nn.Sequential(*ht_reg_branch)

        sz_reg_branch = []
        for _ in range(self.num_reg_fcs):
            sz_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            sz_reg_branch.append(nn.ReLU())
        sz_reg_branch.append(Linear(self.embed_dims, 3))
        sz_reg_branch = nn.Sequential(*sz_reg_branch)

        rot_reg_branch = []
        for _ in range(self.num_reg_fcs):
            rot_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            rot_reg_branch.append(nn.ReLU())
        rot_reg_branch.append(Linear(self.embed_dims, 2))
        rot_reg_branch = nn.Sequential(*rot_reg_branch)

        if self.code_size == 10:
            vel_reg_branch = []
            for _ in range(self.num_reg_fcs):
                vel_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                # reg_branch.append(nn.LayerNorm(self.embed_dims))
                vel_reg_branch.append(nn.ReLU())
            vel_reg_branch.append(Linear(self.embed_dims, 2))
            vel_reg_branch = nn.Sequential(*vel_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.center_reg_branches = _get_clones(center_reg_branch, num_pred)
            self.ht_reg_branches = _get_clones(ht_reg_branch, num_pred)
            self.sz_reg_branches = _get_clones(sz_reg_branch, num_pred)
            self.rot_reg_branches = _get_clones(rot_reg_branch, num_pred)
            if self.code_size == 10:
                self.vel_reg_branches = _get_clones(vel_reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.center_reg_branches = nn.ModuleList(
                [center_reg_branch for _ in range(num_pred)])
            self.ht_reg_branches = nn.ModuleList(
                [ht_reg_branch for _ in range(num_pred)]
            )
            self.sz_reg_branches = nn.ModuleList(
                [sz_reg_branch for _ in range(num_pred)])
            self.rot_branches = nn.ModuleList(
                [rot_reg_branch for _ in range(num_pred)])
            if self.code_size == 10:
                self.vel_reg_branches = nn.ModuleList(
                    [vel_reg_branch for _ in range(num_pred)])

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def forward(self, mlvl_pts_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        batch_size = mlvl_pts_feats[0].size(0)

        #query_embeds = self.query_embedding.weight
        
        lidar_feat = mlvl_pts_feats[0].squeeze(dim=1)
        lidar_feat = self.input_porj(lidar_feat)
        #print('lidar_feat', lidar_feat.size())
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)
        mlvl_pts_feats = [lidar_feat.unsqueeze(dim=1)]
        if self.initialize_by_heatmap:
            dense_heatmap = self.heatmap_head(lidar_feat)

            heatmap = dense_heatmap.detach().sigmoid()
            #print('original heatmap', heatmap.view(batch_size, -1)[..., :20])
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_query]
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            self.query_labels = top_proposals_class
            #query_score = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)
            #query_score = query_score.permute(0, 2, 1)
            # print('query_score', query_score)
            #query_score = torch.nan_to_num(query_score.float())
            
            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())
            query = query_feat + query_cat_encoding
            query = query.permute(2, 0, 1)

            reference_points_2d = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
            #print("reference_points_2d", reference_points_2d)
            #reference_points_2d[:, 0] /= self.x_size
            #reference_points_2d[:, 1] /= self.y_size
            #reference_points = torch.cat((reference_points_2d, torch.rand(batch_size, self.num_query, 1).to(reference_points_2d.device)), dim=-1)


        hs, init_reference, inter_references = self.transformer(
            mlvl_pts_feats,
            #mlvl_img_feats,
            query,
            reference_points_2d,
            reg_branches=self.center_reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
                
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            
            outputs_class = self.cls_branches[lvl](hs[lvl])
            offset = self.center_reg_branches[lvl](hs[lvl])
            height = self.ht_reg_branches[lvl](hs[lvl])
            bbox_sz = self.sz_reg_branches[lvl](hs[lvl])
            rot = self.rot_reg_branches[lvl](hs[lvl])
            if self.code_size == 10:
                vel = self.vel_reg_branches[lvl](hs[lvl])
                tmp = torch.cat((offset[..., :2], bbox_sz[..., :2], height, 
                        bbox_sz[..., 2:3], rot, vel), -1)
            else:
                tmp = torch.cat((offset[..., :2], bbox_sz[..., :2], height, 
                        bbox_sz[..., 2:3], rot), -1)
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            #tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            
            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'dense_heatmap': dense_heatmap,
            #'query_cls_score': query_score,
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size-1]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        if self.initialize_by_heatmap:
            device = labels.device
            #gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
            heatmap = gt_bboxes.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes)):
                width = gt_bboxes[idx][3]
                length = gt_bboxes[idx][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes[idx][0], gt_bboxes[idx][1]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels[idx]], center_int, radius)

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, heatmap)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, heatmap_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, heatmap_list)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    dense_heatmap,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, heatmap_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        heatmap = torch.stack(heatmap_list, 0)

        # heatmap loss
        dense_heatmap = clip_sigmoid(dense_heatmap)
        #print('heatmap', heatmap.size())
        #print('dense_heatmap', dense_heatmap.size())
        loss_heatmap = self.loss_heatmap(dense_heatmap, heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
        
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :self.code_size], normalized_bbox_targets[isnotnan, :self.code_size],
                 bbox_weights[isnotnan, :self.code_size], avg_factor=num_total_pos)
        # loss_bbox_vel = self.loss_bbox(
                # bbox_preds[isnotnan, 8:], normalized_bbox_targets[isnotnan, 8:], bbox_weights[isnotnan, 8:], avg_factor=num_total_pos)
        # loss_bbox = loss_bbox + loss_bbox_vel * 0.2

        #loss_iou = self.loss_iou(
        #    denormalize_bbox(bbox_preds, self.pc_range)[:, :7], bbox_targets[:, :7], bbox_weights[:, 0], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_heatmap
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        dense_heatmap = preds_dicts['dense_heatmap']
        #query_cls_score = preds_dicts['query_cls_score']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_bbox_preds)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        
        #query_cls_score_list = [query_cls_score for _ in range(num_dec_layers)]
        #query_cls_scores = torch.stack(query_cls_score_list)
        dense_heatmap_list = [dense_heatmap for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_heatmap = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            dense_heatmap_list, 
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_heatmap'] = losses_heatmap[-1]
        print('loss_cls', losses_cls[-1])
        print('loss_heatmap', loss_dict['loss_heatmap'])
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size-1)
            if self.code_size == 8:
                bboxes[:, -1] += math.pi
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list