# 文件: mmdet3d/models/fusion_layers/voxel_flow.py
import torch
import torch.nn as nn
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.runner import BaseModule

@FUSION_LAYERS.register_module()
class EffectiveVoxelFlow(BaseModule):
    """

    使用 Z轴位置编码  来打破高度对称性,
    并使用深度可分离3D卷积在虚拟高度上进行高效处理。

    Args:
        c (int): 输入和输出的通道数。
        z (int): 虚拟Z轴（高度）维度, 默认为 4。
        init_cfg (dict, optional): 初始化配置。默认为 None。
    """
    def __init__(self, c, z=4, init_cfg=None):
        super(EffectiveVoxelFlow, self).__init__(init_cfg)
        self.z = z
        self.c = c

        # 1. 关键：可学习的Z轴位置编码
        # 形状为 (1, C, Z, 1, 1)，它将广播到 (B, C, Z, H, W)
        self.z_pos_embed = nn.Parameter(torch.randn(1, c, z, 1, 1))

        # 2. 高效的3D处理层 (深度可分离卷积 + 逐点卷积)
        self.conv_3d = nn.Sequential(
            # 2a. Z轴深度卷积：在每个通道内，独立地混合Z轴信息
            nn.Conv3d(c, c,
                      kernel_size=(3, 1, 1), # Z轴混合
                      padding=(1, 0, 0), # Z轴padding为1, H/W轴为0
                      groups=c, # 深度可分离
                      bias=False),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),

            # 2b. 1x1x1 逐点卷积：混合通道信息
            nn.Conv3d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm3d(c)
            # 注意：这里没有ReLU，让残差可以是负的
        )

        # 3. 最终激活 (在残差相加之后)
        self.relu = nn.ReLU(inplace=True)

        # 4. 压回BEV之前的 1x1x1 卷积 (可选，但通常有好处)
        self.down_conv = nn.Conv3d(c, c, 1)

    def forward(self, bev_feat):
        """
        Args:
            bev_feat (torch.Tensor): 输入 BEV 特征图, shape [B, C, H, W]。

        Returns:
            torch.Tensor: 经过 VF 增强的 BEV 特征图, shape [B, C, H, W]。
        """
        B, C, H, W = bev_feat.shape

        # 1. 升维: (B, C, H, W) -> (B, C, Z, H, W)
        vox = bev_feat.unsqueeze(2).repeat(1, 1, self.z, 1, 1)

        # 2. 注入Z轴位置编码 (打破对称性)
        vox_informed = vox + self.z_pos_embed

        # 3. 3D处理 (在虚拟Z轴上卷积)
        vox_mixed = self.conv_3d(vox_informed)

        # 4. 聚合Z轴 (压回2D)
        # (B, C, Z, H, W) -> (B, C, H, W)
        bev_reproj = self.down_conv(vox_mixed).mean(2)

        # 5. 残差连接 + 最终激活
        return bev_feat + self.relu(bev_reproj)

    # (可选) 添加权重初始化方法
    def init_weights(self):
        super(EffectiveVoxelFlow, self).init_weights()
        # 例如，可以对 z_pos_embed 进行正态初始化
        nn.init.normal_(self.z_pos_embed, mean=0, std=0.02)
        # Conv3d 层通常会自动初始化 (kaiming 或 xavier)