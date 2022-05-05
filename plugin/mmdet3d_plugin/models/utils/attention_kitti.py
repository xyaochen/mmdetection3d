import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from mmcv import ConfigDict, deprecated_api_warning
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init, constant_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils import Transformer
from .attention import inverse_sigmoid

def feature_sampling_3D(mlvl_feats, reference_points, pc_range):
    #reference_points = reference_points.clone()
    #reference_points_rel = reference_points[..., 0:2]
    #print('ref_point_rel', reference_points_rel.size())
    #reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    #reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    #reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 3)
    B, num_query = reference_points.size()[:2]

    #reference_points_rel[..., 1] = (reference_points[..., 0] / pc_range[3]) * 2 - 1
    #reference_points_rel[..., 0] = reference_points[..., 1] / pc_range[4]

    reference_points_rel = reference_points.new_zeros((B, num_query, 2))
    reference_points_rel[..., 0:1] = 2 * reference_points[..., 0:1] - 1
    reference_points_rel[..., 1:2] = 2 * reference_points[..., 1:2] - 1

    sampled_feats = []
    num_points = 1
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_rel_lvl = reference_points_rel.view(B*N, int(num_query/10), 10, 2)
        sampled_feat = F.grid_sample(feat, reference_points_rel_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, num_points).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, 1,  num_points, len(mlvl_feats))
    return sampled_feats



@ATTENTION.register_module()
class FUTR3DCrossAttenLiDARKitti(BaseModule):
    """An attention module used in Deformable-Detr. `Deformable DETR:
    Deformable Transformers for End-to-End Object Detection.
      <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 use_dconv=False,
                 use_level_cam_embed=False,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(FUTR3DCrossAttenLiDARKitti, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = nn.Dropout(weight_dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.use_dconv = use_dconv
        self.use_level_cam_embed = use_level_cam_embed

        self.pts_attention_weights = nn.Linear(embed_dims, 
                                            num_levels*num_points)

        self.pts_output_proj = nn.Linear(embed_dims, embed_dims)


        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
        )

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.pts_output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_feats=None,
                pts_feats=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        # (B, 1, num_query, num_cams, num_points, num_levels)
        pts_attention_weights =  self.pts_attention_weights(query).view(
            bs, 1, num_query, 1, self.num_points, self.num_levels)

        # reference_points: (bs, num_query, 3)
        # output (B, C, num_query, num_cam, num_points, len(lvl_feats))

        pts_output= feature_sampling_3D(
            pts_feats, reference_points, self.pc_range)
        pts_output = torch.nan_to_num(pts_output)
        
        pts_attention_weights = self.weight_dropout(pts_attention_weights.sigmoid())
        pts_output = pts_output * pts_attention_weights
        pts_output = pts_output.sum(-1).sum(-1).sum(-1)
        pts_output = pts_output.permute(2, 0, 1)

        pts_output = self.pts_output_proj(pts_output)

        reference_points_3d = reference_points.clone()
        # (num_query, bs, embed_dims)
        return self.dropout(pts_output) + inp_residual + self.pos_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)



@ATTENTION.register_module()
class FUTR3DCrossAttenCamLiDARKitti(BaseModule):
    """An attention module used in Deformable-Detr. `Deformable DETR:
    Deformable Transformers for End-to-End Object Detection.
      <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 use_dconv=False,
                 use_level_cam_embed=False,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(FUTR3DCrossAttenCamLiDARKitti, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.weight_dropout = nn.Dropout(weight_dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.use_dconv = use_dconv
        self.use_level_cam_embed = use_level_cam_embed
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)
        self.pts_attention_weights = nn.Linear(embed_dims, 
                                            num_levels*num_points)
        self.img_output_proj = nn.Linear(embed_dims, embed_dims)
        self.pts_output_proj = nn.Linear(embed_dims, embed_dims)
        self.modality_fusion_layer = nn.Sequential(
            nn.Linear(2*self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
        )

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.img_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.pts_output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_feats=None,
                pts_feats=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        # (B, 1, num_query, num_cams, num_points, num_levels)
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        pts_attention_weights =  self.pts_attention_weights(query).view(
            bs, 1, num_query, 1, self.num_points, self.num_levels)

        # reference_points: (bs, num_query, 3)
        # output (B, C, num_query, num_cam, num_points, len(lvl_feats))
        reference_points_3d, img_output, mask = feature_sampling(
            img_feats, reference_points, self.pc_range, kwargs['img_metas'])
        img_output = torch.nan_to_num(img_output)
        mask = torch.nan_to_num(mask)

        # TODO: use if else to switch between dynamic conv and weighted sum 

        attention_weights = self.weight_dropout(attention_weights.sigmoid()) * mask
        img_output = img_output * attention_weights
        # output (B, emb_dims, num_query)
        img_output = img_output.sum(-1).sum(-1).sum(-1)
        # output (num_query, B, emb_dims)
        img_output = img_output.permute(2, 0, 1)
        
        img_output = self.img_output_proj(img_output)

        pts_output= feature_sampling_3D(
            pts_feats, reference_points, self.pc_range)
        pts_output = torch.nan_to_num(pts_output)
        
        pts_attention_weights = self.weight_dropout(pts_attention_weights.sigmoid())
        pts_output = pts_output * pts_attention_weights
        pts_output = pts_output.sum(-1).sum(-1).sum(-1)
        pts_output = pts_output.permute(2, 0, 1)

        pts_output = self.pts_output_proj(pts_output)
        #print(img_output.size(), pts_output.size())
        output = torch.cat((img_output, pts_output), dim=2).permute(1, 0, 2)
        output = self.modality_fusion_layer(output).permute(1, 0, 2)

        # (num_query, bs, embed_dims)
        return self.dropout(output) + inp_residual + self.pos_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)


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
    '''
    # inverse rotate, scale and translate
    for i in range(B):
        if 'pcd_trans' in img_metas[i]:
            pcd_trans = torch.from_numpy(img_metas[i]['pcd_trans']).cuda()
            #print('pcd_trans', pcd_trans)
            reference_points[i, :, :3] -= pcd_trans
        if 'pcd_scale_factor' in img_metas[i]:
            pcd_scale_factor = img_metas[i]['pcd_scale_factor']
            #print('pcd_scale', pcd_scale_factor)
            reference_points[i, :, :3] /= pcd_scale_factor
        if 'pcd_rotation' in img_metas[i]:
            pcd_rotation = img_metas[i]['pcd_rotation'].cuda()
            #print('pcd_rotation', pcd_rotation)
            reference_points[i, :, :3] = torch.matmul(reference_points[i], torch.linalg.inv(pcd_rotation))
    '''
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    
    #num_cam = lidar2img.size(1)
    num_cam = 1
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
    #print(img_metas[0]['img_shape'])
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0]
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
        #print(feat.size())
        B, N, C, H, W = feat.size()
        feat_flip = torch.flip(feat, [-1])
        feat = feat.view(B*N, C, H, W)
        # ref_point_cam shape change from (B, num_cam, num_query, 2) to (B*num_cam, num_query/10, 10, 2)
        reference_points_cam_lvl = reference_points_cam.view(B*N, int(num_query/10), 10, 2)
        # sample_feat shape (B*N, C, num_query/10, 10)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # sampled_feat shape (B, C, num_query, N, num_points)
        sampled_feat = sampled_feat.view(B, N, C, num_query, num_points).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  num_points, len(mlvl_feats))
    # ref_point_3d (B, N, num_query, 3)  maks (B, N, num_query, 1)
    return reference_points_3d, sampled_feats, mask