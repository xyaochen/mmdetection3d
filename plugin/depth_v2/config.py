U
    Oʇ`o  �                   @   sV   d dl Z d dlmZ d dlmZmZ G dd� dej�ZG dd� dej�Zd	dd�Z	dS )
�    N)�
ConvModule�build_upsample_layerc                
       sP   e Zd ZdZdddddedd�edd�ed	d�ddf
� fd
d�	Zdd� Z�  ZS )�UpConvBlocka�  Upsample convolution block in decoder for UNet.
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
    �   �   FN�BN)�type�ReLU�
InterpConvc                    s�   t t| ���  |d kstd��|d ks.td��|d| ||||||	|
|d d d�| _|d k	rrt|||||
|d�| _nt||ddd|	|
|d�| _d S )NzNot implemented yet.r   )�in_channels�out_channels�	num_convs�stride�dilation�with_cp�conv_cfg�norm_cfg�act_cfg�dcn�plugins)�cfgr   r   r   r   r   r   r   )�kernel_sizer   �paddingr   r   r   )�superr   �__init__�AssertionError�
conv_blockr   �upsampler   )�selfr   r   �skip_channelsr   r   r   r   r   r   r   r   �upsample_cfgr   r   ��	__class__� �B/home/chenxy/mmdetection3d/plugin/depth_lidar_supervision/utils.pyr   *   sF    ��
�zUpConvBlock.__init__c                 C   s*   | � |�}tj||gdd�}| �|�}|S )zForward function.r   )�dim)r   �torch�catr   )r   �skip�x�outr#   r#   r$   �forward\   s    

zUpConvBlock.forward)�__name__�
__module__�__qualname__�__doc__�dictr   r+   �__classcell__r#   r#   r!   r$   r      s   (�2r   c                       s(   e Zd ZdZ� fdd�Zdd� Z�  ZS )�DepthPredictHeadz�
    1. We use a softplus activation to generate positive depths. 
        The predicted depth is no longer bounded.
    2. The network predicts depth rather than disparity, and at a single scale.
    c                    s$   t t| ���  tj|ddd�| _d S )Nr   )r   )r   r2   r   �nn�Conv2d�conv)r   r   r!   r#   r$   r   l   s    zDepthPredictHead.__init__c                 C   s   | � |�}tj�|�}|S )N)r5   r3   �
functional�softplus)r   r)   �predr#   r#   r$   r+   r   s    
zDepthPredictHead.forward)r,   r-   r.   r/   r   r+   r1   r#   r#   r!   r$   r2   e   s   r2   c           
      C   s�   |dk	r$t �|�}| | } || }n| �� }|d }||  }t �t �|��| }t �t �|�| �| }t �|d | �| }t �t �|d �| �}t �t �t �|�t �| � d �| �}	|||||	fS )zX
    params:
    pred: [N,1,H,W].  torch.Tensor
    gt: [N,1,H,W].     torch.Tensor
    Ng      �?r   )r&   �sum�numel�abs�sqrt�log)
r8   �gt�mask�numZdiff_i�abs_diff�abs_rel�sq_rel�rmse�rmse_logr#   r#   r$   �get_depth_metrics{   s&    

���rF   )N)
r&   �torch.nnr3   �mmcv.cnnr   r   �Moduler   r2   rF   r#   r#   r#   r$   �<module>   s
   _