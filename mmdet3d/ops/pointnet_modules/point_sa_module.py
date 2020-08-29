import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
from typing import List

from mmdet3d.ops import (GroupAll, QueryAndGroup, furthest_point_sample,
                         gather_points)


class PointSAModuleMSG(nn.Module):
    """Point set abstraction module with multi-scale grouping used in
    Pointnets.
    SA包括sampling, grouping和pointNet三个部分
    sampling是使用FPS，grouping是用ball query
    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 num_point: int,#FPS采样点的数量
                 radii: List[float],#ball query的半径
                 sample_nums: List[int],#每个ball query采样的点的数量，由于是multi-scale grouping，所以采样的数目不一样
                 mlp_channels: List[List[int]],# mlp的channel
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod='max',# 池化方式
                 normalize_xyz: bool = False,
                 edge_arg: bool = False):#update 2020/8/28
        super().__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']

        self.num_point = num_point
        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)): #一共ball query多少次
            # 回顾：每个SA都包含sampling(FPS), grouping(ball query)和pointNet(由于ball query得到的每个
            # 下采样点的特征维度是不同的，需要用pointNet来将其变为固定长度)三个部分，需要给定的参数有：
            # sampling阶段的sample_num，ball query阶段的radius(radii)
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    edge_arg=edge_arg)
            else:#当num_point时None的时候，即没有下采样点，没进行FPS
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper) # 储存group用的函数的ModuleList()
            # ball query的输出尺寸应该为BxMxKxC，
            # 其中B是batch size，M是下采样点的个数，K是ball query中最大的近邻点的个数，C是特征维度，
            # 在论文中的表述为(3+d)，包括坐标和point feature
            # SA module就相当于二维卷积网络中的卷积层，用于提取local feature，要是做分类，就在之后加上全连接层，若是
            # 做segmentation或是detection，还需要把它还原到原点云空间去，所以我们需要Feature Propogation，即通过
            # 插值，把local feature还原到原点云中的点上去。

            mlp_spec = mlp_channels[i]
            if use_xyz:
                mlp_spec[0] += 3

            mlp = nn.Sequential() # pointNet部分
            # 关于这里用到的mmcv中的conv module
            """
            这里的convModule包括了卷积层，BN层和激活函数。
            A conv block that bundles conv/norm/activation layers.
            This block simplifies the usage of convolution layers, which are commonly
            used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
            It is based upon three build methods: `build_conv_layer()`,
            `build_norm_layer()` and `build_activation_layer()`.
            Besides, we add some additional features in this module.
            1. Automatically set `bias` of the conv layer.
            2. Spectral norm is supported.
            3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
            supports zero and circular padding, and we add "reflect" padding mode.
            Args:
                in_channels (int): Same as nn.Conv2d.
                out_channels (int): Same as nn.Conv2d.
                kernel_size (int | tuple[int]): Same as nn.Conv2d.
                stride (int | tuple[int]): Same as nn.Conv2d.
                padding (int | tuple[int]): Same as nn.Conv2d.
                dilation (int | tuple[int]): Same as nn.Conv2d.
                groups (int): Same as nn.Conv2d.
                bias (bool | str): If specified as `auto`, it will be decided by the
                    norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
                    False. Default: "auto".
                conv_cfg (dict): Config dict for convolution layer. Default: None,
                    which means using conv2d.
                norm_cfg (dict): Config dict for normalization layer. Default: None.
                act_cfg (dict): Config dict for activation layer.
                    Default: dict(type='ReLU').
                inplace (bool): Whether to use inplace mode for activation.
                    Default: True.
                with_spectral_norm (bool): Whether use spectral norm in conv module.
                    Default: False.
                padding_mode (str): If the `padding_mode` has not been supported by
                    current `Conv2d` in PyTorch, we will use our own padding layer
                    instead. Currently, we support ['zeros', 'circular'] with official
                    implementation and ['reflect'] with our own implementation.
                    Default: 'zeros'.
                order (tuple[str]): The order of conv/norm/activation layers. It is a
                    sequence of "conv", "norm" and "act". Common examples are
                    ("conv", "norm", "act") and ("act", "conv", "norm").
                    Default: ('conv', 'norm', 'act').
            """
            for i in range(len(mlp_spec) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_spec[i], #由此看来，mlp_spec（list）储存的是pointNet中各层的输入和输出，这里的pointNet应该就只是
                        #一系列一维卷积，用来获取fixed feature vector
                        #mlp_channels储存的是各个pointNet的channel，list[list]
                        mlp_spec[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg))
            self.mlps.append(mlp)

    def forward(
        self,
        points_xyz: torch.Tensor,
        features: torch.Tensor = None,
        indices: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features(point) xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        xyz_flipped = points_xyz.transpose(1, 2).contiguous() #这里flip的作用是把points变为Bx3xN，符合函数输入格式
        #调用transpose的时候用contiguous来深拷贝
        if indices is None: #第一层
            indices = furthest_point_sample(points_xyz, self.num_point)# 采样得到的点的indice，因此可以知道num_point是
            #下采样点的数量
        else:
            assert (indices.shape[1] == self.num_point)#保证尺寸是(Bxnum_point)

        new_xyz = gather_points(xyz_flipped, indices).transpose(
            1, 2).contiguous() if self.num_point is not None else None

        for i in range(len(self.groupers)):# 迭代处理每层SA层
            # (B, C, num_point, nsample)，和2DCNN的size一致，可以直接用二维卷积
            new_features = self.groupers[i](points_xyz, new_xyz, features) # 输出每层group的结果
            # 输入是point_xyz: 当前点云的坐标，new_xyz: 下采样点的坐标，features：当前点云的特征
            # 输出格式batch_size*feature*sampling_num*ball_query_num
            # 在这里points_xyz，new_xyz，features会否更新？不会，这不是MRG

            # (B, mlp[-1], num_point, nsample)，mlp[-1]为mlp最后一层的channel数，一维卷积只会改变channel数
            new_features = self.mlps[i](new_features)
            
            if self.pool_mod == 'max':
                # (B, mlp[-1], num_point, 1)
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)])#通过pooling将nsample降为1维
            elif self.pool_mod == 'avg':
                # (B, mlp[-1], num_point, 1)
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], num_point)，去掉一个维度
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices #由于是Multi-scale grouping，改变的是
        # ball query 中sample point的数量，最后用cat把得到的特征串联，得到的是(B, sum_k(mlp_k[-1]), num_point)，
        # 只做一次FPS


class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module used in Pointnets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 num_point: int = None,
                 radius: float = None,
                 num_sample: int = None,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 normalize_xyz: bool = False,
                 edge_arg: bool = False):
        super().__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz,
            edge_arg=False)
