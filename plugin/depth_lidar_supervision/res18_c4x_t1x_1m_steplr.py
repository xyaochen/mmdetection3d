

# Set plugin = True
plugin = True


#model = dict(
#    type='UNet',
#    in_channels=3,
#    out_channels=1,
#    base_channels=16,
#    num_stages=5,
#    strides=(1, 1, 1, 1, 1),
#    enc_num_convs=(2, 2, 2, 2, 2),
#    dec_num_convs=(2, 2, 2, 2),
#    downsamples=(True, True, True, True),
#    norm_cfg=dict(type='BN'),
#    act_cfg=dict(type='ReLU'),
#    upsample_cfg=dict(type='InterpConv'),
#    )
model = dict(
    type='ResDepthModel',
    depth=18,
    strides=(1, 1, 1, 1),
    dilations=(1, 2, 4, 8),
    out_indices=(0, 1, 2, 3),
    base_channels=64,
)


file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'), # filename = results['img_info']['filename']； results['img'] = img
    dict(
        type='Resize',
        img_scale=(675, 1200), # short_edge, long_edge
        multiscale_mode='value',
        keep_ratio=True,),
    dict(type='Normalize', **img_norm_cfg), # normalize will transpose
    dict(type='LoadDepthImage', img_size=(675, 1200), render_type='naive'), # results['seg_fields']
    #dict(type='RandomFlip', flip_ratio=0.5), # if depth -> mask, can resize, flip, rotate
    dict(type='RandomCrop', crop_size=(480, 896), crop_type='absolute', allow_negative_crop=True),
    #dict(type='RandomCrop', crop_size=(200, 200), crop_type='absolute', allow_negative_crop=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'depth_map']),

]
train_pipeline2 = [
    dict(type='LoadImageFromFile'), # filename = results['img_info']['filename']； results['img'] = img
    dict(
        type='Resize',
        img_scale=(896, 480), # w, h; note after reading is (h=900, w=1600)
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='LoadDepthImage', img_size=(480, 896), render_type='naive'), # results['seg_fields']
    dict(type='LoadDepthImage', img_size=(120, 224), render_type='naive'), # results['seg_fields']
    #dict(type='RandomFlip', flip_ratio=0.5), # if depth -> mask, can resize, flip, rotate
    #dict(type='RandomCrop', crop_size=(480, 896), crop_type='absolute'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'depth_map']),

]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='NuscDepthDataset',
        data_path='data/nuscenes/depth_maps/train',
        pipeline=train_pipeline2,
        training=True,
    ),
    val=dict(
        type='NuscDepthDataset',
        data_path='data/nuscenes/depth_maps/train',
        pipeline=train_pipeline2,
        training=False,
    ),
)

checkpoint_config = dict(interval=2)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
#workflow = [('train', 1), ('val', 1)]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.001)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[12, 16],
)
momentum_config = None

# runtime settings
total_epochs = 20
