import torch
import torch.nn as nn
from mmdet3d.models.builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class InstanceBEVFusion(nn.Module):
    def __init__(self, bev_channels, instance_feature_channels, output_channels):
        super(InstanceBEVFusion, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=bev_channels + instance_feature_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fusion_layer(x)