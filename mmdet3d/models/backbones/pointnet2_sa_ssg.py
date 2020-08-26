import torch
from mmcv.runner import load_checkpoint
from torch import nn as nn

from mmdet3d.ops import PointFPModule, PointSAModule
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class PointNet2SASSG(nn.Module):
    """PointNet2 with Single-scale grouping.

    SA包括sampling, grouping和pointNet
    sampling是使用FPS，grouping是用ball query
    FP是把特征从下采样的点插值到原点云上

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
        pool_mod (str): Pool method ('max' or 'avg') for SA modules.
        use_xyz (bool): Whether to use xyz as a part of features.
        normalize_xyz (bool): Whether to normalize xyz with radii in
            each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),#一共四个SA层，分别FPS采样这四个数目的点
                 radius=(0.2, 0.4, 0.8, 1.2),#4个SA层的ball query半径
                 num_samples=(64, 32, 16, 16), #4个SA层ball query的sample num
                 sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                              (128, 128, 256)),#每个SA层最后的pointNet包含3个MLP层，一共4个SA层
                 fp_channels=((256, 256), (256, 256)),#两个FP层，每个FP层有两层
                 norm_cfg=dict(type='BN2d'), #选择BN
                 pool_mod='max', #池化方式
                 use_xyz=True, #是否使用xyz特征
                 normalize_xyz=True): #是否归一化xyz
        super().__init__()

        self.num_sa = len(sa_channels)
        self.num_fp = len(fp_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(fp_channels)
        assert pool_mod in ['max', 'avg']

        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz，要是use_xyz，会在SA层里面再加3
        skip_channel_list = [sa_in_channel]

        # 首先是SA层，共4个
        for sa_index in range(self.num_sa): # SA层的数目
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps # 因为论文表述，得到的特征数是(3+d)
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                PointSAModule(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    use_xyz=use_xyz,
                    pool_mod=pool_mod,
                    normalize_xyz=normalize_xyz))
            skip_channel_list.append(sa_out_channel)#储存每个SA层的输出channel数
            sa_in_channel = sa_out_channel
        # FP层，共2个
        self.FP_modules = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()# pop()返回的是skip_channel_list[-1]
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

    def init_weights(self, pretrained=None):
        """Initialize the weights of PointNet backbone."""
        # 采用预训练模型
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    @staticmethod
    def _split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()#尺寸(B, N, 3)
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()#尺寸(B, C, N)
        else:
            features = None

        return xyz, features

    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of \
                    each fp features.
                - fp_features (list[torch.Tensor]): The features \
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]# batch size和每个batch的点的数量
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()#每行都是1:numpoints，共有batch行

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])# cur_indices是FPS的结果，索引的形式，尺寸(B, M)
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))# dim=1，代表按行看，用cur_indices每行的元素作为sa_indices对应
                #的行的索引，cur_indices必须用long，这里是用作把当前SA的索引和上个SA的索引统一，都还原到原点云的索引上

        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])

        ret = dict(
            fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
        return ret
