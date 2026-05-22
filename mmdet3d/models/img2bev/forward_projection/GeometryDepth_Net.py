import torch, os, cv2, pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from mmcv.runner.dist_utils import master_only
from mmcv.runner import BaseModule
from mmdet3d.models.builder import FUSION_LAYERS
from .modules.Mono_DepthNet_modules import DepthNet2
from .modules.Stereo_Depth_Net_modules import SimpleUnet, convbn_2d, DepthAggregation
from ....utils.depth_tools import generate_guassian_depth_target
from ....utils.depth_tools import get_downsample_depths_torch
from mmdet3d.models.builder import FUSION_LAYERS, build_loss
import matplotlib.pyplot as plt


class DepthVolumeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthVolumeEncoder, self).__init__()
        self.stem = convbn_2d(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.Unet = nn.Sequential(
            SimpleUnet(out_channels)
        )
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        x = self.Unet(x)
        x = self.conv_out(x)
        return x


@FUSION_LAYERS.register_module()
class GeometryDepth_Net(BaseModule):
    def __init__(
            self,
            use_extra_depth=False,
            use_radar_depth=False,
            data_config=None,
            downsample=8,
            input_channels=1280,
            numC_input=512,
            numC_Trans=64,
            cam_channels=27,
            figures_path=None,
            grid_config=None,
            loss_prob_weight=1.00,
            loss_abs_weight=0.001,
            loss_sam2_weight=0.010,
            foreground_loss_alpha=5.0,
            loss_depth_type='bce',
            relative_loss_weight=0.0,
    ):
        super(GeometryDepth_Net, self).__init__()

        self.use_extra_depth = use_extra_depth
        self.use_radar_depth = use_radar_depth
        self.downsample = downsample
        self.input_channels = input_channels
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.cam_channels = cam_channels
        self.grid_config = grid_config
        self.data_config = data_config
        self.alpha = foreground_loss_alpha
        # if no noisy radar engagement, no need to refocus in foreground object
        if not self.use_radar_depth: self.alpha = 0.0

        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.cam_depth_range = self.grid_config['dbound']

        if self.use_radar_depth:
            self.radar_depth_volume_encoder = DepthVolumeEncoder(in_channels=D, out_channels=D)
            self.radar_depth_aggregation = DepthAggregation(input_dims=D, embed_dims=D, out_channels=1)

        if self.use_extra_depth:
            self.extra_depth_volume_encoder = DepthVolumeEncoder(in_channels=D, out_channels=D)
            self.extra_depth_aggregation = DepthAggregation(input_dims=D, embed_dims=D, out_channels=1)

        self.depth_net = DepthNet2(self.input_channels, self.numC_input, self.numC_Trans, self.D,
                                   cam_channels=self.cam_channels)

        self.loss_prob_weight = loss_prob_weight
        self.loss_abs_weight = loss_abs_weight
        self.loss_sam2_weight = loss_sam2_weight
        self.loss_depth_type = loss_depth_type
        self.relative_loss_weight = relative_loss_weight
        if self.relative_loss_weight > 0:
            self.relative_loss = build_loss(
                dict(
                    type='SigmoidRankingLoss',
                )
            )
        self.constant_std = 0.5

    def get_bce_depth_loss(self, depth_labels, depth_preds, radar_depth, img, extra_depth, precise_depth,
                           rangeview_logit):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())

        return depth_loss

    def get_klv_depth_loss(self, depth_labels, depth_preds, rangeview_logit):
        B, bin_size, H, W = depth_preds.shape
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
                                                                             self.downsample, self.cam_depth_range,
                                                                             constant_std=self.constant_std)

        depth_values = depth_values.view(B, H, W)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (
                    depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))

        depth_gaussian_labels = depth_gaussian_labels.view(B, H, W, self.D)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(B, H, W, self.D)

        # saving train depth estimating images
        # self.draw_depth(depth_labels, depth_preds, depth_gaussian_labels, img, extra_depth, precise_depth, radar_depth, show = B)

        depth_gaussian_labels = depth_gaussian_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        pv_foreground_logit = rangeview_logit.squeeze(1)[fg_mask]

        # sparse radar depth make edge bad, thus need stronger depth supervision 'foreground-aware'
        if self.use_radar_depth:
            kl_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='none', log_target=False)
            weighted_kl_loss = kl_loss * (1 + self.alpha * pv_foreground_logit.unsqueeze(-1))
            depth_loss = weighted_kl_loss.sum() / weighted_kl_loss.shape[0]
        else:
            depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean',
                                  log_target=False)

        return depth_loss

    def get_depth_loss(self, depth_labels, depth_preds, precise_depth, rangeview_logit, my_gt_depth=None,
                       img_metas=None):
        sample_idx = img_metas[0]['sample_idx']
        # exit(0)
        if self.loss_depth_type == 'bce':
            depth_loss_prob = self.get_bce_depth_loss(depth_labels, depth_preds, rangeview_logit)

        elif self.loss_depth_type == 'kld':
            depth_loss_prob = self.get_klv_depth_loss(depth_labels, depth_preds, rangeview_logit)

        else:
            pdb.set_trace()

        # extra precise loss of depth
        gt_depths_down = get_downsample_depths_torch(depth_labels, down=self.downsample, processing='min')
        mask = (gt_depths_down > self.cam_depth_range[0]) & (gt_depths_down < self.cam_depth_range[1])
        depth_loss_abs = F.smooth_l1_loss(precise_depth[mask], gt_depths_down[mask])
        '''
        # loss re-weighted
        depth_loss_prob = self.loss_prob_weight * depth_loss_prob
        depth_loss_abs = self.loss_abs_weight * depth_loss_abs
        return dict(depth_loss_prob=depth_loss_prob, depth_loss_abs=depth_loss_abs)
        '''
        depth_loss_sam2=0.0
        if self.loss_abs_weight > 0 and my_gt_depth is not None:
            mask = (my_gt_depth > self.cam_depth_range[0]) & \
                   (my_gt_depth < self.cam_depth_range[1])
            if mask.sum() > 0:
                depth_loss_sam2 = F.smooth_l1_loss(
                    precise_depth.squeeze(1)[mask],
                    my_gt_depth[mask]
                )
        losses = dict(
            depth_loss_prob=self.loss_prob_weight * depth_loss_prob,
            depth_loss_abs=self.loss_abs_weight * depth_loss_abs,
            depth_loss_sam2=self.loss_sam2_weight * depth_loss_sam2
        )

        lidar_presence_mask_orig = (depth_labels > 0).float()

        lidar_presence_mask_down = F.max_pool2d(lidar_presence_mask_orig, kernel_size=self.downsample,
                                                stride=self.downsample)
        lidar_presence_mask = lidar_presence_mask_down.squeeze(1).bool()

        if self.relative_loss_weight > 0 and my_gt_depth is not None:

            pred_depth_for_loss = precise_depth.squeeze(1)
            base_sampling_mask = (my_gt_depth > 0.1) & (my_gt_depth < 65)
            gt_depth_for_loss = torch.clamp(my_gt_depth,
                                            min=self.cam_depth_range[0],
                                            max=self.cam_depth_range[1] - self.cam_depth_range[2])

            # no_lidar_mask = ~lidar_presence_mask

            # B, H, W = my_gt_depth.shape
            # sky_mask = torch.ones_like(my_gt_depth, dtype=torch.bool, device=my_gt_depth.device)
            # sky_height_threshold = int(H * 0.2)
            # sky_mask[:, :sky_height_threshold, :] = False
            # hybrid_weight_map = torch.full_like(my_gt_depth, fill_value=no_lidar_region_weight)
            # hybrid_weight_map[lidar_presence_mask] = with_lidar_region_weight
            segmentation_masks = [meta['segmentation'] for meta in img_metas]

            seg_mask_tensor_bool = torch.from_numpy(np.stack(segmentation_masks)).to(pred_depth_for_loss.device).bool()

            seg_mask_tensor_float = seg_mask_tensor_bool.unsqueeze(1).float()

            downsampled_seg_mask_float = F.max_pool2d(
                seg_mask_tensor_float,
                kernel_size=self.downsample,
                stride=self.downsample
            )
            foreground_mask = (downsampled_seg_mask_float.squeeze(1) > 0).bool()
            mask_for_dilation = foreground_mask.float().unsqueeze(1)

            kernel_size = 5
            padding = (kernel_size - 1) // 2

            dilated_mask_float = F.max_pool2d(
                mask_for_dilation,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )

            dilated_fg_mask = (dilated_mask_float.squeeze(1) > 0).bool()
            background_mask = ~dilated_fg_mask
            edge_sampling_area = dilated_fg_mask & base_sampling_mask
            loss_edge = self.relative_loss(
                pred_depth=pred_depth_for_loss,
                gt_depth=gt_depth_for_loss,
                sampling_mask=edge_sampling_area,
                num_samples=2000
            )
            global_sampling_area = background_mask & base_sampling_mask
            loss_global = self.relative_loss(
                pred_depth=pred_depth_for_loss,
                gt_depth=gt_depth_for_loss,
                sampling_mask=base_sampling_mask,
                num_samples=1000
            )

            w_edge = 1.5
            w_global = 0.5
            loss_relative = (w_edge * loss_edge + w_global * loss_global)
            losses['loss_depth_relative'] = loss_relative * self.relative_loss_weight
            # exit(0)
        return losses

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        # may be different resolution
        if self.training:
            final_dim = self.data_config['final_dim']
        else:
            final_dim = self.data_config['final_dim_test']
        target_H = final_dim[0] // self.downsample
        target_W = final_dim[1] // self.downsample
        assert H // target_H == W // target_W
        down_here = H // target_H

        gt_depths = gt_depths.view(B * N,
                                   H // down_here, down_here,
                                   W // down_here, down_here, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, down_here * down_here)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // down_here, W // down_here)

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / \
                    self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]

        return gt_depths_vals, gt_depths.float()

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        # preparation input
        B, _, _ = rot.shape

        # extrincs: cam2lidar matrix
        # rot: [B, 3, 3]
        # tran: [B, 3]

        # intrincs: cam2img, projection matrix
        # intrin: [B, 4, 4]

        # img-aug, important
        # post_rot: [B, 3, 3]
        # post_tran: [B, 3]

        # bev-aug, test=eye(4)
        # bda: [B, 4, 4]

        mlp_input = torch.stack([
            intrin[:, 0, 0],
            intrin[:, 1, 1],
            intrin[:, 0, 2],
            intrin[:, 1, 2],
            intrin[:, 0, 3],
            intrin[:, 1, 3],
            intrin[:, 2, 3],
            post_rot[:, 0, 0],
            post_rot[:, 0, 1],
            post_tran[:, 0],
            post_rot[:, 1, 0],
            post_rot[:, 1, 1],
            post_tran[:, 1],
            bda[:, 0, 0],
            bda[:, 0, 1],
            bda[:, 1, 0],
            bda[:, 1, 1],
            bda[:, 2, 2],
            bda[:, 0, 3],
            bda[:, 1, 3],
            bda[:, 2, 3],
        ], dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, 3, 1)], dim=-1).reshape(B, -1)  # 12=3x4
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)  # 21+12=33
        mlp_input = mlp_input.to(torch.float32)
        mlp_input = mlp_input.to('cuda') if torch.cuda.is_available() else mlp_input

        return mlp_input

    def forward(self, input, radar_depth, extra_depth, img_metas):

        x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input = input

        B, C, H, W = x.shape
        N = 1

        # mono depth estimation from pure 2D features
        x = self.depth_net(x, mlp_input)
        # exit(0)

        mono_digit = x[:, :self.D, ...]
        mono_volume = self.get_depth_dist(mono_digit)
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]

        # we can of course output here
        depth_volume = mono_volume

        # if we project sparse radar to image for better mono depth estimation
        if self.use_radar_depth:
            _, radar_depth_volume = self.get_downsampled_gt_depth(radar_depth)
            radar_depth_volume = radar_depth_volume.view(B, H, W, -1).permute(0, 3, 1, 2)

            radar_depth_volume = self.radar_depth_volume_encoder(radar_depth_volume)

            radar_depth_volume = self.get_depth_dist(radar_depth_volume)
            depth_volume = self.radar_depth_aggregation(radar_depth_volume, depth_volume)
            depth_volume = self.get_depth_dist(depth_volume)

        # if we use extra depth for better depth estimation
        if self.use_extra_depth:
            _, extra_depth_volume = self.get_downsampled_gt_depth(extra_depth)
            extra_depth_volume = extra_depth_volume.view(B, H, W, -1).permute(0, 3, 1, 2)
            extra_depth_volume = self.extra_depth_volume_encoder(extra_depth_volume)
            extra_depth_volume = self.get_depth_dist(extra_depth_volume)
            depth_volume = self.extra_depth_aggregation(extra_depth_volume, depth_volume)
            depth_volume = self.get_depth_dist(depth_volume)

        # exit(0)
        return img_feat.view(B, N, -1, H, W), depth_volume

