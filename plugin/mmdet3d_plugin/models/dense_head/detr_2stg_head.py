
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
                        
from mmdet.core import (multi_apply, build_assigner, build_sampler, 
                        multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DeformableDETRHead, DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from plugin.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from plugin.mmdet3d_plugin.core.bbox.bbox_ops import center_to_corner_box2d


@HEADS.register_module()
class DeformableFUTR3D2StgHeadCamLiDAR(DETRHead):
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
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 num_cams=6,
                 num_points=7,
                 num_levels=4,
                 per_scene_noise=False,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        #if self.as_two_stage:
        #transformer['as_two_stage'] = self.as_two_stage
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
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_points = num_points
        super(DeformableFUTR3D2StgHeadCamLiDAR, self).__init__(
            *args, transformer=transformer, **kwargs)
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

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred =  self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)

        if self.as_two_stage:
            self.img_attention_weights = nn.Linear(self.embed_dims,
                                           self.num_cams*self.num_levels*self.num_points)
            self.pts_attention_weights = nn.Linear(self.embed_dims,
                                           self.num_levels*self.num_points)
            self.two_stage_fusion = nn.Sequential(
                nn.Linear(2*self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
            )
            
            two_reg_branch = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.code_size), 
            )
            two_cls_branch = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.cls_out_channels), 
            )
            #if self.with_box_refine:
            #    self.two_cls_branches = _get_clones(two_cls_branch, num_pred)
            #    self.two_reg_branches = _get_clones(two_reg_branch, num_pred)
            #else:
            self.two_cls_branches = nn.ModuleList(
                    [two_cls_branch for _ in range(num_pred)])
            self.two_reg_branches = nn.ModuleList(
                    [two_reg_branch for _ in range(num_pred)])

        
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_pts_feats, mlvl_img_feats, img_metas):
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

        query_embeds = self.query_embedding.weight

        if self.per_scene_noise:
            query_pos, query_embeds = torch.split(query_embeds, self.embed_dims , dim=1)
            query_embeds = torch.cat(
                (torch.randn_like(query_embeds), query_embeds), dim=1
            )
        
        hs, init_reference, inter_references = self.transformer(
            mlvl_pts_feats,
            mlvl_img_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
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
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            
            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            if self.as_two_stage:
                num_query, bs, _ = hs[lvl].size()
                img_attention_weights = self.img_attention_weights(hs[lvl]).view(
                    bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
                pts_attention_weights =  self.pts_attention_weights(hs[lvl]).view(
                    bs, 1, num_query, 1, self.num_points, self.num_levels)
                
                # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
                # mask shape (B, 1, num_query, num_cam, num_points, 1)
                sampled_img_feats, mask = bbox_feature_sampling(mlvl_img_feats, tmp, img_metas)
                sampled_pts_feats = bbox_feature_sampling_3D(mlvl_pts_feats, tmp, self.pc_range)
                
                img_attention_weights = img_attention_weights.sigmoid() * mask
                sampled_img_feats = sampled_img_feats * img_attention_weights
                # output (B, emb_dims, num_query)
                sampled_img_feats = sampled_img_feats.sum(-1).sum(-1).sum(-1)
                # output (num_query, B, emb_dims)
                #img_output = sampled_img_feats.permute(2, 0, 1)

                sampled_pts_feats = sampled_pts_feats * pts_attention_weights
                sampled_pts_feats = sampled_pts_feats.sum(-1).sum(-1).sum(-1)
                #pts_output = sampled_pts_feats.permute(2, 0, 1)

                mul_output = torch.cat((sampled_img_feats, sampled_pts_feats), dim=2)
                mul_output = self.two_stage_fusion(mul_output)

                refined_query = hs[lvl] + mul_output

                refined_outputs_class = self.two_cls_branches[lvl](refined_query)
                offset = self.two_reg_branches[lvl](refined_query)
                refined_tmp = tmp + offset

                outputs_classes.append(refined_outputs_class)
                outputs_coords.append(refined_tmp)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
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

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

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
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
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
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

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
                bbox_preds[isnotnan, :self.code_size], 
                normalized_bbox_targets[isnotnan, :self.code_size], 
                bbox_weights[isnotnan, :self.code_size], avg_factor=num_total_pos)
        # loss_bbox_vel = self.loss_bbox(
                # bbox_preds[isnotnan, 8:], normalized_bbox_targets[isnotnan, 8:], bbox_weights[isnotnan, 8:], avg_factor=num_total_pos)
        # loss_bbox = loss_bbox + loss_bbox_vel * 0.2

        #loss_iou = self.loss_iou(
        #    denormalize_bbox(bbox_preds, self.pc_range)[:, :7], bbox_targets[:, :7], bbox_weights[:, 0], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
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
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
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
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list


def bbox_feature_sampling(mlvl_feats, bboxes, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    
    B, num_query = bboxes.size()[:2]
    
    center = torch.cat((bboxes[..., :2], bboxes[..., 4:5]), -1)
    dim_2d = bboxes[..., 2:4]
    height = bboxes[..., 5:6]
    rot_sine = bboxes[..., 6:7]
    rot_cosine = bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    
    corners = center_to_corner_box2d(center[..., :2].view(B*num_query, -1), 
                                    dim_2d.view(B*num_query, -1), rot.view(B*num_query))
    corners = corners.view(B, num_query, -1, 2)
    #print('corners', corners.size())
    front_middle = torch.cat([(corners[..., 0, 0:2] + corners[..., 1, 0:2])/2, center[..., 2:3]], dim=-1)
    back_middle = torch.cat([(corners[..., 2, 0:2] + corners[..., 3, 0:2])/2, center[..., 2:3]], dim=-1)
    left_middle = torch.cat([(corners[..., 0, 0:2] + corners[..., 3, 0:2])/2, center[..., 2:3]], dim=-1)
    right_middle = torch.cat([(corners[..., 1, 0:2] + corners[..., 2, 0:2])/2, center[..., 2:3]], dim=-1) 

    top_middle = center
    top_middle[..., 2:3] += height / 2
    bottom_middle = center
    bottom_middle[..., 2:3] -= height / 2

    num_points = 7
    reference_points = torch.stack([center, top_middle, bottom_middle, front_middle, 
                                    back_middle, left_middle, right_middle], dim=2)
    print('refer' ,reference_points.size())
    # reference_points (B, num_queries, num_points, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    num_cam = lidar2img.size(1)
    # ref_point change to (B, num_cam, num_query, num_points, 4, 1)
    reference_points = reference_points.view(B, 1, num_query, num_points, 4).repeat(1, num_cam, 1, 1, 1).unsqueeze(-1)
    # lidar2img chaneg to (B, num_cam, num_query, num_points, 4, 4)
    lidar2img = lidar2img.view(B, num_cam, 1, 1, 4, 4).repeat(1, 1, num_query, num_points, 1, 1)
    # ref_point_cam change to (B, num_cam, num_query, num_points, 4)
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
    # mask shape (B, 1, num_query, num_cam, num_points, 1)
    mask = mask.view(B, num_cam, 1, num_query, num_points, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []

    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        
        feat = feat.view(B*N, C, H, W)
        # ref_point_cam shape change from (B, num_cam, num_query, num_points, 2) to (B*num_cam, num_query/10, 10, 2)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, num_points, 2)
        # sample_feat shape (B*N, C, num_query, num_points)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # sampled_feat shape (B, C, num_query, N, num_points)
        sampled_feat = sampled_feat.view(B, N, C, num_query, num_points).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, num_points, len(mlvl_feats))
    # maks (B, 1, num_query, num_cam, num_points, 1)
    return sampled_feats, mask

def bbox_feature_sampling_3D(mlvl_feats, bboxes, pc_range):
    B, num_query = bboxes.size()[:2]
    
    center = torch.cat((bboxes[..., :2], bboxes[..., 4:5]), -1)
    dim_2d = bboxes[..., 2:4]
    height = bboxes[..., 5:6]
    rot_sine = bboxes[..., 6:7]
    rot_cosine = bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    
    corners = center_to_corner_box2d(center[..., :2].view(B*num_query, -1), 
                                    dim_2d.view(B*num_query, -1), rot.view(B*num_query))
    corners = corners.view(B, num_query, -1, 2)
    #print('corners', corners.size())
    front_middle = torch.cat([(corners[..., 0, 0:2] + corners[..., 1, 0:2])/2, center[..., 2:3]], dim=-1)
    back_middle = torch.cat([(corners[..., 2, 0:2] + corners[..., 3, 0:2])/2, center[..., 2:3]], dim=-1)
    left_middle = torch.cat([(corners[..., 0, 0:2] + corners[..., 3, 0:2])/2, center[..., 2:3]], dim=-1)
    right_middle = torch.cat([(corners[..., 1, 0:2] + corners[..., 2, 0:2])/2, center[..., 2:3]], dim=-1) 

    top_middle = center
    top_middle[..., 2:3] += height / 2
    bottom_middle = center
    bottom_middle[..., 2:3] -= height / 2

    num_points = 7
    reference_points = torch.stack([center, top_middle, bottom_middle, front_middle, 
                                    back_middle, left_middle, right_middle], dim=2)

    reference_points_rel = reference_points.new_zeros((B, num_query, num_points, 2))
    reference_points_rel[..., 0] = reference_points[..., 0] / pc_range[3]
    reference_points_rel[..., 1] = reference_points[..., 1] / pc_range[4]

    sampled_feats = []
    
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_rel_lvl = reference_points_rel.view(B*N, num_query, num_points, 2)
        sampled_feat = F.grid_sample(feat, reference_points_rel_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, num_points).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, 1,  num_points, len(mlvl_feats))
    return sampled_feats