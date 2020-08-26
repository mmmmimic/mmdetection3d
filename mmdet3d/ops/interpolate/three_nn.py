import torch
from torch.autograd import Function
from typing import Tuple

from . import interpolate_ext


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, target: torch.Tensor,
                source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the top-3 nearest neighbors of the target set from the source
        set.

        Args:
            target (Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.
                M是下采样点的数量
                B是batch大小？
                3是采样点的3个坐标

        Returns:
            Tensor: shape (B, N, 3), L2 distance of each point in target
                set to their corresponding nearest neighbors.
        """
        assert target.is_contiguous()#应该是为了CUDA后续的处理，要保证相邻的元素在内存中也是相邻的
        assert source.is_contiguous()

        B, N, _ = target.size()
        m = source.size(1) #下采样点的数量
        dist2 = torch.cuda.FloatTensor(B, N, 3) #每个点都对应3个最近的距离
        idx = torch.cuda.IntTensor(B, N, 3) #每个点对应三个下采样点的索引

        interpolate_ext.three_nn_wrapper(B, N, m, target, source, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply
