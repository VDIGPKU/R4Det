import inspect
import Pyro4
import uuid
import traceback
from mmdet.models import HEADS
import warnings
import torch.distributed as dist
import torch, copy, time, os, mmcv
import torch.nn as nn
from torch import device
from torch.nn import functional as F
import numpy as np
from shapely.geometry import Polygon, box, Point
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmcv.runner.dist_utils import master_only
from mmdet.models import DETECTORS
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models import builder
from mmdet3d.models.builder import FUSION_LAYERS, NECKS
from mmdet3d.models.detectors.mvx_faster_rcnn import MVXFasterRCNN
from mmdet3d.core import bbox3d2result, show_multi_modality_result, LiDARInstance3DBoxes
from ...datasets.structures.bbox import HorizontalBoxes, bbox2roi
from ...utils.visualization import draw_bev_pts_bboxes, draw_paper_bboxes
from ...utils.visualization import custom_draw_lidar_bbox3d_on_img
from mmdet.models import build_head
from mmdet3d.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet3d.models.dense_heads import Anchor3DHead
from mmcv.ops import box_iou_rotated
from mmcv.runner import force_fp32
def _calculate_bev_iou(gt_boxes, pred_boxes):
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return torch.tensor([], device=gt_boxes.device)
    gt_bev = gt_boxes[:, [0, 1, 3, 4, 6]]
    pred_bev = pred_boxes[:, [0, 1, 3, 4, 6]]

    iou_matrix = box_iou_rotated(gt_bev, pred_bev)
    return iou_matrix


class IFGDFusionModule(nn.Module):
    def __init__(self, bev_channels, instance_channels):
        super(IFGDFusionModule, self).__init__()
        self.bev_channels = bev_channels
        self.instance_channels = instance_channels
        if bev_channels != instance_channels:
            self.project_layer = nn.Sequential(
                nn.Conv2d(instance_channels, instance_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(instance_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.project_layer = nn.Identity()
        self.conv_gamma = nn.Conv2d(self.instance_channels, self.bev_channels, 1)
        self.conv_beta = nn.Conv2d(self.instance_channels, self.bev_channels, 1)

        nn.init.constant_(self.conv_gamma.weight, 0)
        nn.init.constant_(self.conv_gamma.bias, 1.0)
        nn.init.constant_(self.conv_beta.weight, 0)
        nn.init.constant_(self.conv_beta.bias, 0)
        self.gate_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, RC_BEV, E_features, S_BEV):

        B, C_bev, W, H = RC_BEV.shape
        N_total, C_inst, _, _ = E_features.shape
        # (N_total, C_inst, H_pool, W_pool) -> (N_total, C_inst)
        E_pooled = E_features.mean(dim=[-1, -2])
        # (N_total, C_inst) -> (N_total, C_inst, 1, 1) -> (N_total, C_inst)
        E_proj = self.project_layer(E_pooled.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        #  S_BEV (B, N, H, W) -> (B, N, W, H)
        S_BEV_permuted = S_BEV.permute(0, 1, 3, 2).contiguous()
        T=0.01
        A_prob = F.softmax(S_BEV_permuted / T, dim=1)  # (B, N_total, W, H)
        # (B, N_total, W*H) -> (B, W*H, N_total)
        A_flat = A_prob.view(B, N_total, -1).permute(0, 2, 1)
        # (N_total, C_inst) -> (B, N_total, C_inst)
        E_proj_batch = E_proj.unsqueeze(0).expand(B, -1, -1)
        # (B, W*H, N_total) @ (B, N_total, C_inst) -> (B, W*H, C_inst)
        E_BEV_flat = torch.bmm(A_flat, E_proj_batch)
        # (B, W*H, C_inst) -> (B, C_inst, W, H)
        E_BEV = E_BEV_flat.permute(0, 2, 1).view(B, C_inst, W, H)
        # (B, C_inst, W, H) -> (B, C_bev, W, H)
        gamma_BEV = self.conv_gamma(E_BEV)
        beta_BEV = self.conv_beta(E_BEV)
        F_fixed = RC_BEV * gamma_BEV + beta_BEV
        G_bg_logits = S_BEV_permuted.sum(dim=1, keepdim=True)  # (B, 1, W, H)
        G_bg = torch.sigmoid(self.gate_conv(G_bg_logits))  # (B, 1, W, H)
        F_Final = (1.0 - G_bg) * RC_BEV + G_bg * F_fixed
        return F_Final

@HEADS.register_module()
class DynamicAnchor3DHead(Anchor3DHead):
    def __init__(self,
                 instance_channels,
                 *args, **kwargs):
        super(DynamicAnchor3DHead, self).__init__(*args, **kwargs)
        self.instance_channels = instance_channels
        self.bev_channels = self.in_channels
        self.cls_gamma_gen = nn.Conv2d(self.instance_channels, self.bev_channels, 1)
        self.cls_beta_gen = nn.Conv2d(self.instance_channels, self.bev_channels, 1)
        self.reg_gamma_gen = nn.Conv2d(self.instance_channels, self.bev_channels, 1)
        self.reg_beta_gen = nn.Conv2d(self.instance_channels, self.bev_channels, 1)

    def init_weights(self):

        nn.init.constant_(self.cls_gamma_gen.weight, 0)
        nn.init.constant_(self.cls_gamma_gen.bias, 0)
        nn.init.constant_(self.cls_beta_gen.weight, 0)
        nn.init.constant_(self.cls_beta_gen.bias, 0)
        nn.init.constant_(self.reg_gamma_gen.weight, 0)
        nn.init.constant_(self.reg_gamma_gen.bias, 0)
        nn.init.constant_(self.reg_beta_gen.weight, 0)
        nn.init.constant_(self.reg_beta_gen.bias, 0)
    def forward(self, feats, E_BEV=None):
        x = feats[0]  # (B, C_bev, W, H)
        if E_BEV is not None:
            gamma_cls = self.cls_gamma_gen(E_BEV)
            beta_cls = self.cls_beta_gen(E_BEV)
            feat_for_cls = x * (1.0 + gamma_cls) + beta_cls  # FiLM
            gamma_reg = self.reg_gamma_gen(E_BEV)
            beta_reg = self.reg_beta_gen(E_BEV)
            feat_for_reg = x * (1.0 + gamma_reg) + beta_reg  # FiLM
        else:
            feat_for_cls = x
            feat_for_reg = x
            gamma_cls = None
        cls_score = self.conv_cls(feat_for_cls)
        bbox_pred = self.conv_reg(feat_for_reg)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(feat_for_reg)
        cls_score_list = [cls_score]
        bbox_pred_list = [bbox_pred]
        dir_cls_pred_list = [dir_cls_pred]
        if self.training:
            if E_BEV is not None and gamma_cls is not None:
                return cls_score_list, bbox_pred_list, dir_cls_pred_list, (gamma_cls, E_BEV)
            else:
                return cls_score_list, bbox_pred_list, dir_cls_pred_list
        else:
            return cls_score_list, bbox_pred_list, dir_cls_pred_list

class DummySamplingResult:
    def __init__(self, num_rois): self.pos_bboxes = torch.zeros((num_rois, 4))


@DETECTORS.register_module()
class R4Det(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""
    def __init__(self,
                 mask_save_dir=None,
                 bev_h_=160,
                 bev_w_=160,
                 img_channels=256,
                 rad_channels=384,
                 num_classes=3,
                 num_in_height=8,
                 use_depth_supervision=True,
                 use_props_supervision=True,
                 use_box3d_supervision=True,
                 use_msk2d_supervision=True,
                 use_backward_projection=False,
                 use_sa_radarnet=False,
                 use_grid_mask=False,
                 freeze_images=True,
                 freeze_depths=False,
                 freeze_radars=False,
                 camera_stream='LSS',
                 point_cloud_range=None,
                 grid_config=None,
                 img_norm_cfg=None,
                 # ablation settings
                 backward_ablation=None,
                 focusradardepth_ablation=None,
                 painting_ablation=None,
                 # model config
                 depth_net=None,
                 rangeview_foreground=None,
                 img_view_transformer=None,
                 RCFusion=None,
                 debug_internal_tensors=True,
                 proposal_layer=None,
                 backward_projection=None,
                 meta_info=None,
                 img_rpn_head=None,
                 img_roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 instance_feature_fusion_layer=None,
                 temporal_fusion=None,
                 **kwargs):
        super(R4Det, self).__init__(train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        HEADS.module_dict['StandardRoIHead'] = StandardRoIHead
        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir:
            mmcv.mkdir_or_exist(self.mask_save_dir)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = build_head(img_rpn_head_)
        if img_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg is not None else None
            img_roi_head.update(train_cfg=rcnn_train_cfg)
            img_roi_head.update(test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = build_head(img_roi_head)

        self.roi_head = self.img_roi_head
        self.rpn_head = self.img_rpn_head
        if self.pts_voxel_encoder:
            # self.pts_dim = self.pts_voxel_encoder.in_channels
            self.pts_dim = kwargs['pts_voxel_encoder']['in_channels']
        else:
            self.pts_dim = None
            warnings.warn("pts_voxel_encoder is None, pts_dim cannot be determined.")
        self.visualize_on_test = True
        self.bev_h_ = bev_h_
        self.bev_w_ = bev_w_
        self.img_channels = img_channels
        self.rad_channels = rad_channels
        self.num_classes = num_classes
        self.num_in_height = num_in_height
        self.freeze_images = freeze_images
        self.freeze_depths = freeze_depths
        self.freeze_radars = freeze_radars
        self.lift_method = camera_stream
        self.point_cloud_range = point_cloud_range
        self.grid_config = grid_config
        self.img_norm_cfg = img_norm_cfg
        self.use_grid_mask = use_grid_mask
        self.use_depth_supervision = use_depth_supervision
        self.use_props_supervision = use_props_supervision
        self.use_box3d_supervision = use_box3d_supervision
        self.use_msk2d_supervision = use_msk2d_supervision
        self.use_backward_projection = use_backward_projection

        self.use_sa_radarnet = use_sa_radarnet
        self.use_radar_depth = depth_net['use_radar_depth']
        self.use_extra_depth = depth_net['use_extra_depth']
        self.backward_ablation = backward_ablation
        self.focusradardepth_ablation = focusradardepth_ablation
        self.painting_ablation = painting_ablation
        self.RCFusion = RCFusion
        self.meta_info = meta_info
        self.figures_path = meta_info['figures_path']
        self.project_name = meta_info['project_name']
        if 'vod' in self.project_name.lower(): self.dataset_type = 'VoD'
        if 'tj4d' in self.project_name.lower(): self.dataset_type = 'TJ4D'

        # other papa for convenience
        self.xbound = self.grid_config['xbound']
        self.ybound = self.grid_config['ybound']
        self.zbound = self.grid_config['zbound']
        self.bev_grid_shape = [bev_h_, bev_w_]
        self.bev_cell_size = [(self.xbound[1] - self.xbound[0]) / bev_h_, (self.ybound[1] - self.ybound[0]) / bev_w_]
        self.voxel_size = [self.grid_config['xbound'][2], self.grid_config['ybound'][2], self.grid_config['zbound'][2]]
        self.backward_use_pv_logits = self.backward_ablation['pv_logits']
        self.backward_use_depth_prob = self.backward_ablation['depth_prob']
        self.painting_use_pv_logits = self.painting_ablation['pv_logits']
        self.painting_use_depth_prob = self.painting_ablation['depth_prob']
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        self.xlim, self.ylim = [x_min, x_max], [y_min, y_max]

        # vanilla model and loss settings
        if self.lift_method == 'OFT': pass
        if self.lift_method == 'LSS':
            depth_net.update(figures_path=self.figures_path)
            self.depth_net = FUSION_LAYERS.build(depth_net) if depth_net else None
            img_view_transformer.update(num_in_height=self.num_in_height)
            self.img_view_transformer = FUSION_LAYERS.build(img_view_transformer)
            self.downsample = self.depth_net.downsample
        self.rangeview_foreground = NECKS.build(rangeview_foreground) if (
                    rangeview_foreground and use_msk2d_supervision) else None
        self.cross_attention = FUSION_LAYERS.build(RCFusion)
        self.proposal_layer_former = FUSION_LAYERS.build(proposal_layer) if (
                    proposal_layer and use_props_supervision) else None
        if self.use_backward_projection:  # bev latter means after RCFusion (use paint BEV fusion or backward projection)
            self.proposal_layer_latter = FUSION_LAYERS.build(proposal_layer) if (
                        proposal_layer and use_props_supervision) else None
        else:
            self.proposal_layer_latter = None
        self.backward_projection = FUSION_LAYERS.build(backward_projection) if (
                    backward_projection and use_backward_projection) else None
        self.temporal_fusion = None
        if temporal_fusion:
            self.temporal_fusion = builder.build_fusion_layer(temporal_fusion)
        # init weights and freeze if needed
        self.init_flexible_modules()
        self.init_weights()
        if self.freeze_images: self.freeze_img_model()
        if self.freeze_radars: self.freeze_pts_model()
        self.record_fps = {'num': 0, 'time': 0}
        self.init_visulization()
        self.ifgd_fusion = IFGDFusionModule(
            bev_channels=self.img_channels,
            instance_channels=self.img_channels
        )
        self.head_downsampler = nn.AvgPool2d(kernel_size=4, stride=4)
        self.debug_internal_tensors = debug_internal_tensors
        fpn_file_path = inspect.getfile(self.img_neck.__class__)
        self.debug_vis_gt_vs_pred = True
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None


    @torch.no_grad()
    @master_only
    def _save_mask_img(self, mask_list, img_meta):
        original_filename = os.path.basename(img_meta.get('filename', 'unknown_file.jpg'))
        img_index = os.path.splitext(original_filename)[0]
        output_dir_for_image = os.path.join(self.mask_save_dir, img_index)
        os.makedirs(output_dir_for_image, exist_ok=True)
        total_saved_count = 0
        for class_idx, masks_for_class in enumerate(mask_list):
            if not isinstance(masks_for_class, list):
                continue
            for instance_idx_in_class, sub_mask in enumerate(masks_for_class):
                mask_to_save = sub_mask.astype(np.uint8) * 255
                if np.sum(mask_to_save) == 0:
                    continue
                save_path = os.path.join(output_dir_for_image,
                                         f'class_{class_idx}_instance_{instance_idx_in_class}.png')
                mmcv.imwrite(mask_to_save, save_path)
                total_saved_count += 1



    def _build_bev_instance_map(self,
                                img_metas,
                                precise_depth,
                                sampling_results,
                                pred_masks,
                                pos_rois,
                                img,
                                N_total
                                ):
        batch_size = len(img_metas)
        device = precise_depth.device
        S_BEV = torch.zeros(batch_size, N_total, self.bev_h_, self.bev_w_, device=device)
        depth_sigma = 1.0
        depth_range_meters = depth_sigma * 4
        num_samples_per_ray = 10

        num_pos_per_img = [len(res.pos_bboxes) for res in sampling_results]
        pred_masks_list = pred_masks.split(num_pos_per_img, 0)
        pos_rois_list = pos_rois.split(num_pos_per_img, 0)
        for i in range(batch_size):
            img_meta = img_metas[i]
            depth_map = precise_depth[i, 0]
            pred_masks_per_img = pred_masks_list[i]
            pos_rois_per_img = pos_rois_list[i]
            if pred_masks_per_img.shape[0] == 0:
                continue
            global_idx_offset = sum(num_pos_per_img[:i])
            cam2lidar = torch.inverse(torch.tensor(img_meta['lidar2cam'], device=device, dtype=torch.float32))
            intrins_matrix = torch.tensor(img_meta['cam2img'], device=device, dtype=torch.float32)
            aug_post_rot = torch.tensor(img_meta['cam_aware'][3][:2, :2], device=device, dtype=torch.float32)
            aug_post_tran = torch.tensor(img_meta['cam_aware'][4][:2], device=device, dtype=torch.float32)
            aug_intrins_matrix = intrins_matrix.clone()
            aug_intrins_matrix[:2, :2] = aug_post_rot @ aug_intrins_matrix[:2, :2]
            aug_intrins_matrix[:2, 2] = aug_post_rot @ aug_intrins_matrix[:2, 2] + aug_post_tran

            aug_h, aug_w = img[i].shape[-2:]
            depth_h, depth_w = depth_map.shape
            scale_w_ratio = depth_w / aug_w
            scale_h_ratio = depth_h / aug_h

            scale_matrix = torch.tensor([[scale_w_ratio, 0, 0], [0, scale_h_ratio, 0], [0, 0, 1]], device=device,
                                        dtype=torch.float32)
            intrins = scale_matrix @ aug_intrins_matrix[:3, :3]
            for idx in range(pred_masks_per_img.shape[0]):
                global_idx = global_idx_offset + idx
                roi_box = pos_rois_per_img[idx, 1:]
                mask_template = pred_masks_per_img[idx]
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
                roi_w = (roi_x2 - roi_x1).round().int()
                roi_h = (roi_y2 - roi_y1).round().int()

                if roi_w < 1 or roi_h < 1: continue
                mask_in_roi = F.interpolate(mask_template.unsqueeze(0).unsqueeze(0).float(),
                                            size=(roi_h.item(), roi_w.item()),
                                            mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                if pred_masks.max() > 1.0:
                    binary_mask_in_roi = mask_in_roi.sigmoid() > 0.5
                else:
                    binary_mask_in_roi = mask_in_roi > 0.5
                coords = torch.where(binary_mask_in_roi)
                if coords[0].numel() == 0:
                    continue

                y_in_roi, x_in_roi = coords
                if y_in_roi.numel() == 0: continue

                abs_y = roi_y1 + y_in_roi
                abs_x = roi_x1 + x_in_roi
                valid_abs_mask = (abs_x >= 0) & (abs_x < aug_w) & (abs_y >= 0) & (abs_y < aug_h)
                if not valid_abs_mask.any():
                    continue
                abs_x = abs_x[valid_abs_mask]
                abs_y = abs_y[valid_abs_mask]
                pixels_y = (abs_y * scale_h_ratio).long().clamp(0, depth_h - 1)
                pixels_x = (abs_x * scale_w_ratio).long().clamp(0, depth_w - 1)

                predicted_depths = depth_map[pixels_y, pixels_x]
                depth_samples_offset = torch.linspace(-depth_range_meters / 2, depth_range_meters / 2,
                                                      num_samples_per_ray, device=device)
                depth_samples = predicted_depths.unsqueeze(1) + depth_samples_offset.unsqueeze(0)
                depth_samples = depth_samples.clamp(min=self.grid_config['dbound'][0])
                gaussian_weights = torch.exp(
                    -0.5 * torch.pow((depth_samples - predicted_depths.unsqueeze(1)) / depth_sigma, 2))

                all_depth_samples, all_pixels_x, all_pixels_y = depth_samples.view(-1), pixels_x.unsqueeze(1).expand(-1,
                                                                                                                     num_samples_per_ray).reshape(
                    -1), pixels_y.unsqueeze(1).expand(-1, num_samples_per_ray).reshape(-1)

                points_cam_z = all_depth_samples
                points_cam_x = (all_pixels_x.float() - intrins[0, 2]) * points_cam_z / intrins[0, 0]
                points_cam_y = (all_pixels_y.float() - intrins[1, 2]) * points_cam_z / intrins[1, 1]
                points_cam_hom = torch.stack([points_cam_x, points_cam_y, points_cam_z, torch.ones_like(points_cam_z)],
                                             dim=0)
                lidar_points = (cam2lidar @ points_cam_hom)[:3, :].T
                bev_x, bev_y = ((lidar_points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]).long(), (
                        (lidar_points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]).long()

                valid_mask = (bev_x >= 0) & (bev_x < self.bev_h_) & (bev_y >= 0) & (bev_y < self.bev_w_)

                if valid_mask.any():
                    valid_bev_x, valid_bev_y = bev_x[valid_mask], bev_y[valid_mask]
                    valid_weights = gaussian_weights.view(-1)[valid_mask]
                    valid_scores = valid_weights

                    unique_coords, inverse_idx = torch.unique(torch.stack([valid_bev_x, valid_bev_y], dim=1),
                                                              return_inverse=True, dim=0)
                    summed_scores = torch.zeros((unique_coords.shape[0]), device=device)
                    summed_scores.index_add_(0, inverse_idx, valid_scores)
                    point_counts = torch.bincount(inverse_idx, minlength=unique_coords.shape[0])
                    averaged_scores = summed_scores / point_counts.clamp(min=1)
                    S_BEV[i, global_idx, unique_coords[:, 0], unique_coords[:, 1]] = averaged_scores

        return S_BEV


    def init_flexible_modules(self):
        if self.use_sa_radarnet:
            self.voxelpainting_points, self.voxel_coords = self.generate_pillar_ref_points(self.num_in_height)
            in_channels = self.num_in_height * self.img_channels
            self.adaptive_collapse_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.rad_channels, kernel_size=1),
                BasicBlock(self.rad_channels, self.rad_channels))
            self.PaintBEVFusion = nn.Sequential(
                BasicBlock(self.rad_channels * 2, self.rad_channels,
                           downsample=nn.Conv2d(self.rad_channels * 2, self.rad_channels, 3, 1, 1)),
                BasicBlock(self.rad_channels, self.rad_channels))

    def init_visulization(self):
        self.SAVE_INTERVALS = 250  # 500
        self.vis_time_box3d = 0
        self.vis_time_bev2d = 0
        self.vis_time_bevnd = 0
        self.vis_time_range = 0
        self.vis_time_point = 0
        self.mean = np.array(self.img_norm_cfg['mean'])
        self.std = np.array(self.img_norm_cfg['std'])
        self.figures_path_det3d_test = os.path.join(self.figures_path, 'test', 'det3d')
        self.figures_path_bev2d_test = os.path.join(self.figures_path, 'test', 'bev_mask')
        self.figures_path_bevnd_test = os.path.join(self.figures_path, 'test', 'bev_feats')
        self.figures_path_range_test = os.path.join(self.figures_path, 'test', 'range')
        self.figures_path_point_test = os.path.join(self.figures_path, 'test', 'point')
        self.figures_path_det3d_train = os.path.join(self.figures_path, 'train', 'det3d')
        self.figures_path_bev2d_train = os.path.join(self.figures_path, 'train', 'bev_mask')
        self.figures_path_bevnd_train = os.path.join(self.figures_path, 'train', 'bev_feats')
        self.figures_path_range_train = os.path.join(self.figures_path, 'train', 'range')
        self.figures_path_point_train = os.path.join(self.figures_path, 'train', 'point')
        os.makedirs(self.figures_path_det3d_test, exist_ok=True)
        os.makedirs(self.figures_path_bev2d_test, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_test, exist_ok=True)
        os.makedirs(self.figures_path_range_test, exist_ok=True)
        os.makedirs(self.figures_path_point_test, exist_ok=True)
        os.makedirs(self.figures_path_det3d_train, exist_ok=True)
        os.makedirs(self.figures_path_bev2d_train, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_train, exist_ok=True)
        os.makedirs(self.figures_path_range_train, exist_ok=True)
        os.makedirs(self.figures_path_point_train, exist_ok=True)

    # model parameter freezing or not
    def freeze_img_model(self):
        """freeze image backbone and neck for fusion"""
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.lift_method == 'LSS' and self.freeze_depths:
            for param in self.depth_net.parameters():
                param.requires_grad = False

    def freeze_pts_model(self):
        """freeze radar backbone and neck for pretrain"""
        if self.pts_voxel_encoder:
            for param in self.pts_voxel_encoder.parameters():
                param.requires_grad = False
        if self.pts_middle_encoder:
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
        if self.pts_backbone:
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
        if self.pts_neck is not None:
            for param in self.pts_neck.parameters():
                param.requires_grad = False
        if self.with_pts_bbox:
            for param in self.pts_bbox_head.parameters():
                param.requires_grad = False

    # feature pre-extraction
    def generate_pillar_ref_points(self, num_in_height):
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        voxel_x, voxel_y, voxel_z = self.voxel_size

        # Calculate the center points for the grid
        x_centers = torch.arange(x_min + voxel_x / 2, x_max, voxel_x)
        y_centers = torch.arange(y_min + voxel_y / 2, y_max, voxel_y)
        z_step = (z_max - z_min) / num_in_height
        z_centers = torch.arange(z_min + z_step / 2, z_max, z_step)
        assert x_centers.shape[0] == self.bev_h_
        assert y_centers.shape[0] == self.bev_w_

        # Create a mesh grid for x, y, z
        xv, yv, zv = torch.meshgrid(x_centers, y_centers, z_centers)
        # Stack the grid coordinates
        ref_points = torch.stack((xv, yv, zv), dim=-1)  # shape: (H, W, Z, 3)
        # indices
        hv, wv, zv = torch.meshgrid(torch.arange(self.bev_h_), torch.arange(self.bev_w_), torch.arange(num_in_height))
        idx = torch.arange(self.bev_h_ * self.bev_w_ * num_in_height)
        voxel_coords = torch.cat([hv.reshape(-1, 1), wv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        return ref_points, voxel_coords

    def extract_pts_feat(self, pts, img_metas):
        """Extract features of raw points."""
        if not self.with_pts_backbone: return None
        batch_size = len(pts)
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors, )
        #batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        self.instance_feature_cache = {}
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    # NOTE: core model here, processing multi-modality feats

    def extract_feat(self, points, img, img_metas, prev_bev_feats=None,is_valid_mask=None,feat_or_dict=0):
        """Extract features from images and points."""
        # preparation of camera-geo-aware input
        if img.dim() == 3 and img.size(0) == 3: img = img.unsqueeze(0)
        B, C, H, W = img.shape
        if not isinstance(points, list): points = [points]

        img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, cam_aware, \
            img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask, final_lidar2img = self.preprocessing_information(
            img_metas, img.device)

        img_inputs = [img, cam_aware[0], cam_aware[1], cam_aware[2], cam_aware[3], cam_aware[4], bda_rot]
        img, rots, trans, intrins, post_rots, post_trans, bda = img_inputs[0:7]
        matrix = torch.eye(4).to(final_lidar2img.device)
        matrix[0, 0] = matrix[0, 0] / self.downsample
        matrix[1, 1] = matrix[1, 1] / self.downsample
        projection = matrix @ final_lidar2img
        start_time = time.time()
        img_feats = self.extract_img_feat(img, img_metas)
        end_time = time.time()
        step1_time = end_time - start_time
        if self.lift_method == 'LSS':
            mlp_input = self.depth_net.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
            geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
            view_trans_inputs = [rots, trans, intrins, post_rots, post_trans, bda]
            cam_params_list = [
                [rots[i:i + 1], trans[i:i + 1], intrins[i:i + 1], post_rots[i:i + 1], post_trans[i:i + 1], bda[i:i + 1]]
                for i in range(img.shape[0])]
            index = self.downsample // 4 - 1
            h, w = img.shape[2] // self.downsample, img.shape[3] // self.downsample
            align_feats = [F.interpolate(feat, (h, w), mode='bilinear', align_corners=True) for feat in img_feats]
            align_feats = torch.cat(align_feats, dim=1)
            start_time = time.time()
            context, depth = self.depth_net([align_feats] + geo_inputs, radar_depth, depth_comple, img_metas)
            end_time = time.time()
            img_bev_feats = self.img_view_transformer(context, depth, view_trans_inputs)
            img_bev_feats = img_bev_feats.mean(-1)  # B, C, bev_h_, bev_w_
            img_bev_feats = img_bev_feats.permute(0, 1, 3, 2).contiguous()
            cam_depth_range = self.grid_config['dbound']
            raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(depth.device)
            precise_depth = torch.sum(raw_depth.view(1, -1, 1, 1) * depth, dim=1).unsqueeze(1)
        else:
            context, depth, precise_depth = None, None, None
        if self.lift_method == 'OFT': pass
        if self.lift_method == None: img_bev_feats = pts_bev_feats
        step2_time = end_time - start_time
        if self.rangeview_foreground is not None and self.use_msk2d_supervision and context is not None:
            start_time = time.time()
            rangeview_logit = self.rangeview_foreground(context.squeeze(1))
            end_time = time.time()
            rangeview_logit_sigmoid = rangeview_logit.sigmoid()
            focus_weight = 1.0 * precise_depth / self.point_cloud_range[3]
            mask_reweighted = (1 - focus_weight) * rangeview_logit_sigmoid + focus_weight * torch.ones_like(
                rangeview_logit_sigmoid).to(context.device)
            min_vals = torch.min(mask_reweighted.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
            max_vals = torch.max(mask_reweighted.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
            mask_reweighted = (mask_reweighted - min_vals) / (max_vals - min_vals)
            rangeview_logit_sigmoid = mask_reweighted
        else:
            rangeview_logit = None
        step3_time = end_time - start_time
        if self.use_sa_radarnet:
            radar_decorate_img_feats = context.squeeze(1)
            pts_bev_feats = self.extract_pts_feat(points, img_metas)[0]
            h, w, z = self.voxelpainting_points.shape[:3]
            points_vis = [
                torch.cat([points[i], torch.zeros((points[i].shape[0], self.img_channels), device=context.device)],
                          dim=-1) for i in range(B)]

            input_points = [self.voxelpainting_points.reshape(-1, 3).to(context.device) for _ in range(B)]
            start_time = time.time()
            all_decorated_points = self.voxelpainting_depth_aware(radar_decorate_img_feats, rangeview_logit.sigmoid(),
                                                                  depth, input_points, projection)
            end_time = time.time()

            feature_sum = [torch.sum(all_decorated_points[i][:, 5:], dim=1) for i in range(B)]
            topk_indice = [torch.topk(feature_sum[i], 200)[1] for i in range(B)]
            decorated_points_vis = [all_decorated_points[i][topk_indice[i]] for i in range(B)]
            self.draw_pts_completion(img_metas, decorated_points_vis, points_vis, gt_bboxes_3d, gt_labels_3d,
                                     plot_mode='context_voxelpainting',
                                     points_type='voxelpainting')
            all_decorated_points = [all_decorated_points[i][:, self.pts_dim:] for i in range(B)]
            paint_bev = [x.view(1, h, w, z, -1) for x in all_decorated_points]
            paint_bev = torch.cat(paint_bev, dim=0)  # B H W Z C
            paint_bev = paint_bev.view(B, h, w, -1).permute(0, 3, 2, 1).contiguous()  # B C*Z H W
            paint_bev = self.adaptive_collapse_conv(paint_bev)
            pts_bev_feats_origin = pts_bev_feats
            if self.dataset_type == 'VoD': paint_bev = 0.01 * paint_bev
            pts_bev_feats = self.PaintBEVFusion(torch.cat([paint_bev, pts_bev_feats], dim=1))
            end_time = time.time()
        else:  # without any painting
            pts_bev_feats = self.extract_pts_feat(points, img_metas)[0]
            paint_bev = torch.zeros_like(pts_bev_feats).to(context.device)
            pts_bev_feats_origin = pts_bev_feats
        step4_time = end_time - start_time

        start_time = time.time()
        bev_feats = self.cross_attention(img_bev_feats, pts_bev_feats)
        end_time = time.time()
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()
        assert bev_feats.shape[2] == self.bev_h_
        assert bev_feats.shape[3] == self.bev_w_
        step5_time = end_time - start_time

        if self.proposal_layer_former is not None and self.use_props_supervision:
            bev_mask_logit_former = self.proposal_layer_former(bev_feats)
        else:
            bev_mask_logit_former = None
        if self.temporal_fusion is not None:
            if prev_bev_feats is not None or feat_or_dict == 1:
                if prev_bev_feats is not None:
                    prev_bev_feats_device = prev_bev_feats.to(bev_feats.device)
                    bev_feats_cache = bev_feats
                    bev_feats = self.temporal_fusion(bev_feats, prev_bev_feats_device)
                    if is_valid_mask is not None:
                        is_valid_mask_dev = is_valid_mask.to(bev_feats.device)
                        mask_expanded = is_valid_mask_dev[:, None, None, None]
                        bev_feats = torch.where(mask_expanded, bev_feats, bev_feats_cache)
            else:
                return bev_feats
        if self.backward_projection is not None and self.lift_method == 'LSS':
            bev_mask = torch.ones((B, 1, self.bev_h_, self.bev_w_), dtype=torch.bool).to(img.device)
            search_img_feats = img_feats[index]
            search_img_feats = search_img_feats * rangeview_logit_sigmoid if self.backward_use_pv_logits else search_img_feats
            search_img_feats = search_img_feats.unsqueeze(1)
            depth_dist = torch.full(depth.shape, 1.0 / depth.shape[1]).to(
                context.device) if not self.backward_use_depth_prob else depth
            start_time = time.time()
            bev_feats_refined = self.backward_projection(
                imgs=img,
                mlvl_feats=[search_img_feats],
                proposal=bev_mask,
                cam_params=cam_params_list,
                lss_bev=bev_feats,
                img_metas=img_metas,
                mlvl_dpt_dists=[depth_dist.unsqueeze(1)],
                backward_bev_mask_logit=torch.zeros_like(bev_feats).to(context.device))
            end_time = time.time()
            bev_feats_refined = bev_feats_refined.permute(0, 2, 1).view(B, self.img_channels, self.bev_h_,
                                                                        self.bev_w_).contiguous()
        else:
            bev_feats_refined = bev_feats
        step6_time = end_time - start_time
        if self.proposal_layer_latter is not None and self.use_props_supervision:
            bev_mask_logit_latter = self.proposal_layer_latter(bev_feats_refined)
        else:
            bev_mask_logit_latter = None
        bev_mask_logit = {'former': bev_mask_logit_former, 'latter': bev_mask_logit_latter}
        bev_feats_refined = bev_feats_refined.permute(0, 1, 3, 2).contiguous()
        step_all_time = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time
        self.recording_fps(step_all_time)

        return dict(img_feats=img_feats,
                    pts_feats=[bev_feats_refined],
                    aux_feats_point=[pts_bev_feats],
                    aux_feats_image=[img_bev_feats],
                    pd_depths=depth,
                    gt_depths=gt_depths,
                    rd_depths=radar_depth,
                    gt_bev_mask=gt_bev_mask,
                    bev_mask_logit=bev_mask_logit,
                    bbox_Mask=bbox_Mask,
                    segmentation=segmentation,
                    rangeview_logit=rangeview_logit,
                    depth_comple=depth_comple,
                    precise_depth=precise_depth)

    def voxelpainting_depth_aware(self, context, pv_logits, depth_logits, points, lidar2img, temperature=1.0):
        B, _, H, W = pv_logits.shape
        device = lidar2img.device
        cam_depth_range = self.grid_config['dbound']
        painted_points = []
        bev_h, bev_w = self.bev_h_, self.bev_w_
        bev_x_min, bev_y_min, bev_z_min, bev_x_max, bev_y_max, bev_z_max = self.point_cloud_range
        context = context * temperature  # enlarge

        for i in range(B):
            # preparation
            pts = points[i][:, :3]
            pts_hom = torch.cat((pts, torch.ones((pts.shape[0], 1), device=device)), dim=1)  # (N, 4)
            img_pts = torch.matmul(lidar2img[i], pts_hom.t()).t()  # (N, 4)
            img_pts[:, :2] = img_pts[:, :2] / img_pts[:, 2:3]
            depth_values = img_pts[:, 2]

            valid_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
            img_pts_int = img_pts[:, :2].long()

            # Initialize the feature tensor with zeros,
            context_features = torch.zeros((pts.shape[0], context.shape[1] + self.pts_dim), device=device,
                                           dtype=context.dtype)
            pts_norm = (pts - torch.tensor([bev_x_min, bev_y_min, bev_z_min], device=device)) \
                       / torch.tensor([bev_x_max - bev_x_min, bev_y_max - bev_y_min, bev_z_max - bev_z_min],
                                      device=device)
            # context_features[:, :3] = pts_norm
            context_features[:, :3] = pts  # keep same as radar points

            # begin valid point decorated
            valid_img_pts_int = img_pts_int[valid_mask]
            valid_depth_values = depth_values[valid_mask]
            # for grid sample for sub pixel
            valid_img_pts_norm = img_pts[:, :2][valid_mask].clone()
            valid_img_pts_norm[:, 0] = (valid_img_pts_norm[:, 0] / (W - 1)) * 2 - 1
            valid_img_pts_norm[:, 1] = (valid_img_pts_norm[:, 1] / (H - 1)) * 2 - 1

            # Extract context features for valid image points
            if self.dataset_type == 'VoD':
                valid_context_features = F.grid_sample(context[i].unsqueeze(0),
                                                       valid_img_pts_norm.unsqueeze(0).unsqueeze(1), align_corners=True)
                valid_context_features = valid_context_features.squeeze(0).squeeze(1).permute(1, 0)  # (num_points, C)
            if self.dataset_type == 'TJ4D':
                valid_context_features = context[i, :, valid_img_pts_int[:, 1], valid_img_pts_int[:, 0]]
                valid_context_features = valid_context_features.permute(1, 0)  # (num_points, C)

            # Get corresponding depth_logits & Weight context_features using the log-transformed depth probabilities
            if self.painting_use_depth_prob:
                depth_probs = depth_logits[i, :, valid_img_pts_int[:, 1], valid_img_pts_int[:, 0]]
                if self.dataset_type == 'VoD': power_exponent = 1
                if self.dataset_type == 'TJ4D': power_exponent = 2
                power_depth_probs = depth_probs ** power_exponent
                power_depth_probs /= power_depth_probs.sum(dim=0, keepdim=True)  # Normalize
                depth_probs = power_depth_probs
                if self.dataset_type == 'VoD':
                    depth_indices = ((valid_depth_values - cam_depth_range[0]) / cam_depth_range[2])
                    lower_indices = torch.floor(depth_indices).long()
                    upper_indices = torch.ceil(depth_indices).long()
                    lower_indices = torch.clamp(lower_indices, 0, depth_probs.shape[0] - 1)
                    upper_indices = torch.clamp(upper_indices, 0, depth_probs.shape[0] - 1)
                    upper_weight = depth_indices - lower_indices.float()
                    lower_weight = 1 - upper_weight
                    lower_prob_values = depth_probs[lower_indices, range(lower_indices.shape[0])]
                    upper_prob_values = depth_probs[upper_indices, range(upper_indices.shape[0])]
                    depth_prob_values = lower_weight * lower_prob_values + upper_weight * upper_prob_values
                if self.dataset_type == 'TJ4D':
                    depth_indices = ((valid_depth_values - cam_depth_range[0]) / cam_depth_range[2]).long()
                    depth_indices = torch.clamp(depth_indices, 0, depth_probs.shape[0] - 1)
                    depth_prob_values = depth_probs[depth_indices, torch.arange(depth_indices.shape[0], device=device)]
                # re-weight decorated features
                valid_context_features = valid_context_features * depth_prob_values.unsqueeze(1)

            # Assign the computed features to the correct positions
            indices = torch.nonzero(valid_mask).squeeze(1).to(device)
            add_feature = torch.cat(
                (torch.zeros((indices.size(0), self.pts_dim), device=device), valid_context_features), dim=1)
            context_features.index_add_(0, indices, add_feature)
            painted_points.append(context_features)

        return painted_points

    # train and evaluating process
    @staticmethod
    def _create_low_res_gt_depth(gt_depths_tensor, target_h, target_w):
        if gt_depths_tensor.shape[0] != 1:
            gt_depths_tensor = gt_depths_tensor[:1]

        device = gt_depths_tensor.device
        gt_depth_map = gt_depths_tensor.squeeze()
        source_h, source_w = gt_depth_map.shape

        valid_mask = gt_depth_map > 0
        ys, xs = torch.where(valid_mask)
        depth_values = gt_depth_map[valid_mask]

        if len(depth_values) == 0:
            return torch.zeros((1, 1, target_h, target_w), device=device)

        scale_h = target_h / source_h
        scale_w = target_w / source_w

        target_ys = (ys * scale_h).long().clamp(0, target_h - 1)
        target_xs = (xs * scale_w).long().clamp(0, target_w - 1)
        sorted_indices = torch.argsort(depth_values, descending=True)
        depth_values_sorted = depth_values[sorted_indices]
        target_ys_sorted = target_ys[sorted_indices]
        target_xs_sorted = target_xs[sorted_indices]

        low_res_gt_map = torch.zeros((target_h, target_w), device=device)
        low_res_gt_map[target_ys_sorted, target_xs_sorted] = depth_values_sorted

        return low_res_gt_map.unsqueeze(0).unsqueeze(0)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_labels=None,
                    gt_bboxes=None, **kwargs):
        """Test function without augmentaiton."""
        self.prev_bev_feats = None
        outs_pts = None
        if len(img_metas) != 1: img_metas = [img_metas]
        prev_points = points[0]
        points = points[1]
        prev_img = img[0, ...]
        img = img[1, ...]
        prev_img_metas = [meta[0] for meta in img_metas]
        img_metas = [meta[1] for meta in img_metas]
        prev_gt_bboxes_3d = [gt[0] for gt in gt_bboxes_3d]
        gt_bboxes_3d = [gt[1] for gt in gt_bboxes_3d]
        prev_gt_labels_3d = [gt[0] for gt in gt_labels_3d]
        gt_labels_3d = [gt[1] for gt in gt_labels_3d]
        prev_gt_labels = [gt[0] for gt in gt_labels] if gt_labels is not None else None
        prev_gt_bboxes = [gt[0] for gt in gt_bboxes] if gt_bboxes is not None else None
        gt_labels = [gt[1] for gt in gt_labels] if gt_labels is not None else None
        gt_bboxes = [gt[1] for gt in gt_bboxes] if gt_bboxes is not None else None
        #gt_masks = [gt[1] for gt in gt_masks] if gt_masks is not None else None
        is_valid_mask = torch.tensor([meta['is_prev_frame_valid'] for meta in prev_img_metas], device=img.device)
        prev_bev_feats = None
        for i in range(len(prev_img_metas)):
            prev_img_metas[i]['gt_labels'] = prev_gt_labels[i]
            prev_img_metas[i]['gt_bboxes'] = HorizontalBoxes(prev_gt_bboxes[i], in_mode='xyxy')
            prev_img_metas[i]['gt_bboxes_3d'] = prev_gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            prev_img_metas[i]['gt_labels_3d'] = prev_gt_labels_3d[i]
        if is_valid_mask.any():
            with torch.no_grad():
                raw_prev_bev_feats = self.extract_feat(prev_points, prev_img, prev_img_metas, prev_bev_feats=None,is_valid_mask=is_valid_mask,feat_or_dict=0)
                prev_bev_feats = raw_prev_bev_feats * is_valid_mask[:, None, None, None]
        if gt_bboxes_3d is not None:
            for i in range(len(img_metas)):
                img_metas[i]['gt_labels'] = gt_labels[i]
                img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas,prev_bev_feats=prev_bev_feats,is_valid_mask=is_valid_mask,feat_or_dict=1)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        precise_depth = feature_dict.get('precise_depth')
        instance_features_test = None
        instance_rois_test = None
        pred_masks_test_logits = None
        det_labels_test = None
        sampling_results_test = None

        if self.with_rpn and self.with_roi_head and self.img_roi_head.with_mask:

            if hasattr(self.img_roi_head, 'simple_test_with_intermediate'):

                img_feats_list = list(img_feats) if isinstance(img_feats, (tuple, list)) else [img_feats]
                proposal_list_2d = self.img_rpn_head.simple_test_rpn(img_feats_list, img_metas)

                bbox_results_2d, mask_results_2d, intermediate_2d = \
                    self.img_roi_head.simple_test_with_intermediate(
                        img_feats_list, proposal_list_2d, img_metas, rescale=rescale, return_intermediate=True
                    )

                if intermediate_2d is not None:
                    instance_features_test = intermediate_2d.get('instance_features')
                    pred_masks_test_logits = intermediate_2d.get('pred_masks_logits')
                    instance_rois_test = intermediate_2d.get('rois')  # RoIs used for mask
                    det_labels_test = intermediate_2d.get('det_labels')  # Corresponding labels
                    indexed_pred_masks = None
                    if pred_masks_test_logits is not None and det_labels_test is not None:
                        if pred_masks_test_logits.shape[0] == det_labels_test.shape[0]:
                            N_det = pred_masks_test_logits.shape[0]
                            indexed_pred_masks = pred_masks_test_logits[
                                                 torch.arange(N_det, device=pred_masks_test_logits.device),
                                                 det_labels_test.long(),
                                                 :, :]
                    if instance_rois_test is not None and instance_rois_test.shape[0] > 0:
                        num_rois_per_img = []
                        for img_idx in range(len(img_metas)):
                            num = torch.sum(instance_rois_test[:, 0] == img_idx).item()
                            num_rois_per_img.append(num)
                        sampling_results_test = [DummySamplingResult(n) for n in num_rois_per_img]
                    else:
                        sampling_results_test = [DummySamplingResult(0) for _ in range(len(img_metas))]
                else:
                    sampling_results_test = [DummySamplingResult(0) for _ in range(len(img_metas))]
            else:
                sampling_results_test = [DummySamplingResult(0) for _ in range(len(img_metas))]
        else:
            sampling_results_test = [DummySamplingResult(0) for _ in range(len(img_metas))]

        pts_feats_refined_list = pts_feats
        if instance_features_test is not None and instance_features_test.shape[0] > 0 and self.ifgd_fusion:
            S_BEV = self._build_bev_instance_map(
                img_metas=img_metas,
                precise_depth=precise_depth,
                sampling_results=sampling_results_test,
                pred_masks=indexed_pred_masks.detach(),
                pos_rois=instance_rois_test,
                img=img,
                N_total=instance_features_test.shape[0]
            )
            RC_BEV = pts_feats_refined_list[0]  # (B, C_bev, W, H)
            E_features = instance_features_test
            fused_bev_feat = self.ifgd_fusion(RC_BEV, E_features, S_BEV)
            pts_feats = [fused_bev_feat]


        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox and self.use_box3d_supervision:  # pts means 3D detection
            bbox_pts, outs_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if img_feats:  # and self.with_img_bbox:  # img means 2D detection
            results = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            bbox_img, mask_img = zip(*results)
            for result_dict, img_bbox, img_mask in zip(bbox_list, bbox_img, mask_img):
                result_dict['img_bbox'] = img_bbox



        # visualization for test stage
        threshold = 0.3
        if gt_bboxes_3d is not None and self.use_box3d_supervision:
            if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
            if not isinstance(points, list): points = [points]
            self.draw_gt_pred_figures_3d(points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=outs_pts)
        else: # vanilla testing method
            self.vis_time_box3d += 1
            if self.vis_time_box3d % self.SAVE_INTERVALS == 0:
                figures_path_det3d = self.figures_path_det3d_test
                input_img = np.array(img.cpu()).transpose(1,2,0)
                input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
                pred_bboxes_3d = bbox_pts[0]['boxes_3d']
                pred_scores_3d = bbox_pts[0]['scores_3d']
                pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu')
                proj_mat = img_metas[0]["final_lidar2img"] # update lidar2img
                img_name = img_metas[0]['filename'].split('/')[-1].split('.')[0]
                # project 3D bboxes to image and get show figures
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'

        return bbox_list


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      gt_masks=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      bev_semantic_mask=None,
                      my_gt_depth=None,
                      **kwargs):
        prev_points = [p[0] for p in points]
        points = [p[1] for p in points]
        prev_img = img[:, 0, ...]
        img = img[:, 1, ...]
        prev_img_metas = [meta[0] for meta in img_metas]
        img_metas = [meta[1] for meta in img_metas]
        prev_gt_bboxes_3d = [gt[0] for gt in gt_bboxes_3d]
        gt_bboxes_3d = [gt[1] for gt in gt_bboxes_3d]
        prev_gt_labels_3d = [gt[0] for gt in gt_labels_3d]
        gt_labels_3d = [gt[1] for gt in gt_labels_3d]
        prev_gt_labels = [gt[0] for gt in gt_labels] if gt_labels is not None else None
        prev_gt_bboxes = [gt[0] for gt in gt_bboxes] if gt_bboxes is not None else None
        prev_gt_bboxes_ignore = [gt[0] for gt in gt_bboxes_ignore] if gt_bboxes_ignore is not None else None
        gt_labels = [gt[1] for gt in gt_labels] if gt_labels is not None else None
        gt_bboxes = [gt[1] for gt in gt_bboxes] if gt_bboxes is not None else None
        gt_masks = [gt[1] for gt in gt_masks] if gt_masks is not None else None
        gt_bboxes_ignore = [gt[1] for gt in gt_bboxes_ignore] if gt_bboxes_ignore is not None else None
        prev_my_gt_depth = my_gt_depth[0]
        my_gt_depth = my_gt_depth[1]
        if bev_semantic_mask is not None and isinstance(bev_semantic_mask, list):
            prev_bev_semantic_mask = bev_semantic_mask[0]
            bev_semantic_mask = bev_semantic_mask[1]

        if img_depth is not None and isinstance(img_depth, list):
            prev_img_depth = img_depth[0]
            img_depth = img_depth[1]

        if proposals is not None and isinstance(proposals, list):
            prev_proposals = proposals[0]
            proposals = proposals[1]
        is_valid_mask = torch.tensor([meta['is_prev_frame_valid'] for meta in prev_img_metas], device=img.device)
        prev_bev_feats = None
        for i in range(len(prev_img_metas)):
            prev_img_metas[i]['gt_labels'] = prev_gt_labels[i]
            prev_img_metas[i]['gt_bboxes'] = HorizontalBoxes(prev_gt_bboxes[i], in_mode='xyxy')
            prev_img_metas[i]['gt_bboxes_3d'] = prev_gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            prev_img_metas[i]['gt_labels_3d'] = prev_gt_labels_3d[i]
        if self.temporal_fusion is not None and is_valid_mask.any():
            with torch.no_grad():
                raw_prev_bev_feats = self.extract_feat(prev_points, prev_img, prev_img_metas, prev_bev_feats=None,
                                                       is_valid_mask=is_valid_mask, feat_or_dict=0)
                prev_bev_feats = raw_prev_bev_feats * is_valid_mask[:, None, None, None]

        # preparation for loss caculation
        for i in range(len(img_metas)):
            img_metas[i]['gt_labels'] = gt_labels[i]
            img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas,prev_bev_feats=prev_bev_feats,is_valid_mask=is_valid_mask,feat_or_dict=1)
        # feature_dict = torch.load(load_path)

        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        gt_depths = feature_dict['gt_depths']
        pd_depths = feature_dict['pd_depths']
        gt_bev_mask = feature_dict['gt_bev_mask']
        bev_mask_logit = feature_dict['bev_mask_logit']
        bbox_Mask = feature_dict['bbox_Mask']
        segmentation = feature_dict['segmentation']
        rangeview_logit = feature_dict['rangeview_logit']
        precise_depth = feature_dict['precise_depth']
        # compute for all losses
        losses = dict()

        # img_feats = feature_dict.get('img_feats')
        instance_features = None

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.img_rpn)
            rpn_losses, proposal_list = self.img_rpn_head.forward_train(
                img_feats,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if self.img_roi_head.with_bbox or self.img_roi_head.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.img_roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.img_roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in img_feats])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.img_roi_head.with_bbox:
            bbox_results = self.img_roi_head._bbox_forward_train(img_feats, sampling_results,
                                                                 gt_bboxes, gt_labels,
                                                                 img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.img_roi_head.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self.img_roi_head._mask_forward_train(img_feats, sampling_results,
                                                                 bbox_results['bbox_feats'],
                                                                 gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            instance_features = mask_results.get('mask_feats')

            raw_mask_pred = mask_results.get('mask_pred')

            if pos_rois is not None and pos_rois.shape[0] > 0 and raw_mask_pred is not None:
                pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
                N = raw_mask_pred.shape[0]
                pred_masks = raw_mask_pred[torch.arange(N, device=raw_mask_pred.device), pos_labels, :, :]

        instance_features_for_fusion = None
        indexed_pred_masks_for_fusion = None
        rois_for_fusion = None
        sampling_results_for_fusion = [DummySamplingResult(0) for _ in range(len(img_metas))]

        if self.with_rpn and self.with_roi_head and self.img_roi_head.with_mask and \
                hasattr(self.img_roi_head, 'simple_test_with_intermediate'):
            with torch.no_grad():
                proposal_list_inf = self.img_rpn_head.simple_test_rpn(img_feats, img_metas)
            _b, _m, intermediate_2d = self.img_roi_head.simple_test_with_intermediate(
                img_feats, proposal_list_inf, img_metas, return_intermediate=True
            )

            if intermediate_2d is not None:
                instance_features_for_fusion = intermediate_2d.get('instance_features')
                pred_masks_logits_for_fusion = intermediate_2d.get('pred_masks_logits')
                rois_for_fusion = intermediate_2d.get('rois')
                det_labels_for_fusion = intermediate_2d.get('det_labels')
                if pred_masks_logits_for_fusion is not None and det_labels_for_fusion is not None and \
                        pred_masks_logits_for_fusion.shape[0] == det_labels_for_fusion.shape[0]:
                    N_det = pred_masks_logits_for_fusion.shape[0]
                    indexed_pred_masks_for_fusion = pred_masks_logits_for_fusion[
                                                    torch.arange(N_det,
                                                                 device=pred_masks_logits_for_fusion.device),
                                                    det_labels_for_fusion.long(), :, :
                                                    ]
                if rois_for_fusion is not None and rois_for_fusion.shape[0] > 0:
                    num_rois_per_img = [torch.sum(rois_for_fusion[:, 0] == i).item() for i in
                                        range(len(img_metas))]
                    sampling_results_for_fusion = [DummySamplingResult(n) for n in num_rois_per_img]

        pts_feats_refined_list = pts_feats
        if instance_features_for_fusion is not None and instance_features_for_fusion.shape[0] > 0 and self.ifgd_fusion and indexed_pred_masks_for_fusion is not None:
            precise_depth = feature_dict['precise_depth']

            S_BEV = self._build_bev_instance_map(
                img_metas=img_metas,
                precise_depth=precise_depth,
                sampling_results=sampling_results_for_fusion,
                pred_masks=indexed_pred_masks_for_fusion.detach(),
                pos_rois=rois_for_fusion,
                img=img,
                N_total=instance_features_for_fusion.shape[0]
            )

            RC_BEV = pts_feats_refined_list[0]  # (B, C_bev, W, H)
            E_features = instance_features_for_fusion
            fused_bev_feat = self.ifgd_fusion(RC_BEV, E_features, S_BEV)
            pts_feats = [fused_bev_feat]
        outs_pts = None
        if self.use_box3d_supervision and gt_bboxes_3d is not None:
            losses_pts, outs_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d,
                                                          img_metas,gt_bboxes_ignore, bev_semantic_mask)
            losses.update(losses_pts)
        if self.use_depth_supervision and self.depth_net is not None:

            loss_depth = self.depth_net.get_depth_loss(gt_depths, pd_depths, precise_depth,
                                                       rangeview_logit.sigmoid().detach(), my_gt_depth=my_gt_depth,
                                                       img_metas=img_metas)  # .detach()
            losses.update(loss_depth)
        if self.use_props_supervision is not None and self.proposal_layer_former is not None:
            if bev_mask_logit['former'] is not None:

                losses_proposal_former = self.proposal_layer_former.get_bev_mask_loss(gt_bev_mask,
                                                                                      bev_mask_logit['former'])
                losses_proposal_former = {f"{key}_former": value for key, value in losses_proposal_former.items()}

                losses.update(losses_proposal_former)
        if self.use_props_supervision is not None and self.proposal_layer_latter is not None:
            if bev_mask_logit['latter'] is not None:

                losses_proposal_latter = self.proposal_layer_latter.get_bev_mask_loss(gt_bev_mask,
                                                                                      bev_mask_logit['latter'])
                losses_proposal_latter = {f"{key}_latter": value for key, value in losses_proposal_latter.items()}
                losses.update(losses_proposal_latter)
        if self.use_msk2d_supervision is not None and self.rangeview_foreground is not None:
            losses_proposal = self.rangeview_foreground.get_range_view_mask_loss(bbox_Mask, segmentation,
                                                                                 rangeview_logit)
            losses.update(losses_proposal)
        return losses


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          bev_semantic_mask=None,
                          ):


        outs = self.pts_bbox_head(pts_feats)

        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs,
            gt_bboxes_ignore=gt_bboxes_ignore,
            bev_semantic_mask=bev_semantic_mask
        )

        # losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
        return losses, outs


    # preprocessing for data and others
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            if res.shape[0] == 0:
                num_features = res.shape[1]
                dummy_point = torch.zeros((1, num_features), dtype=res.dtype, device=res.device)
                dummy_point[0, :3] = -1000.0
                res = dummy_point
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    def preprocessing_information(self, batch_img_metas, device):
        if self.training:
            # all important informations
            batch_size = len(batch_img_metas)
            # gt lidar instances_3d, list of InstancesData
            gt_bboxes_3d = [img_meta['gt_bboxes_3d'] for img_meta in batch_img_metas]
            gt_labels_3d = [img_meta['gt_labels_3d'] for img_meta in batch_img_metas]
            gt_bboxes_2d = [img_meta['gt_bboxes'] for img_meta in batch_img_metas]
            gt_labels_2d = [img_meta['gt_labels'] for img_meta in batch_img_metas]

            # cam_aware: rot, tran, intrin, post_rot, post_tran, _, cam2lidar, focal_length, baseline
            # print(f"DEBUG: batch_img_metas keys: {batch_img_metas.keys()}")
            # if 'cam_aware' not in batch_img_metas:
            #    print("!!! ERROR: 'cam_aware' is missing in batch_img_metas right before access !!!")

            cam_aware = [img_meta['cam_aware'] for img_meta in batch_img_metas]
            merged_tensors = [None] * len(cam_aware[0])
            for i in range(len(cam_aware[0])):
                component = [x[i] for x in cam_aware]
                merged_tensors[i] = torch.stack(component, dim=0)
            cam_aware = merged_tensors
            cam_aware = [x.to(device) for x in cam_aware]

            # img_aug_matrix: 4x4 martix of combined post_rot&post_tran of IMG_AUG
            img_aug_matrix = [img_meta['img_aug_matrix'] for img_meta in batch_img_metas]
            img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
            img_aug_matrix = img_aug_matrix.to(device)
            # lidar_aug_matrix same as bda_rot: 4x4 martix of combined post_rot&post_tran of BEV_AUG
            if 'lidar_aug_matrix' in batch_img_metas[0]:
                lidar_aug_matrix = [img_meta['lidar_aug_matrix'] for img_meta in batch_img_metas]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32)
                bda_rot = [img_meta['bda_rot'] for img_meta in batch_img_metas]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32)
            else:
                lidar_aug_matrix = torch.eye(4).unsqueeze(0).repeat(len(batch_img_metas), 1, 1)
                bda_rot = lidar_aug_matrix
            lidar_aug_matrix = lidar_aug_matrix.to(device)
            bda_rot = bda_rot.to(device)
            # create gt_depths from LiDAR data, already processing with IMG_AUG, no need with BEV_AUG
            gt_depths = [img_meta['gt_depths'] for img_meta in batch_img_metas if 'gt_depths' in img_meta]
            gt_depths = torch.stack(gt_depths).unsqueeze(1)  # B, 1, H, W
            gt_depths = gt_depths.to(device)

            # generate_bev_mask
            gt_bboxes_3d_filtered = [gt_bboxes_3d[i][gt_labels_3d[i] != -1] for i in
                                     range(batch_size)]  # filter out the ignored labels
            gt_bev_mask = self.generate_bev_mask(gt_bboxes_3d_filtered, batch_size, device, occ_threshold=0.3)  # B H W
            gt_bev_mask = gt_bev_mask.to(device)

            # re-organize clearly to create NOW lidar2img for project convenience
            batch_img_metas = self.reorganize_lidar2img(batch_img_metas)
            calib = []
            for sample_idx in range(batch_size):
                mat = batch_img_metas[sample_idx]['final_lidar2img']
                mat = torch.Tensor(mat).to(device)
                calib.append(mat)
            final_lidar2img = torch.stack(calib)

            # preprocessed seg_mask, pre_inferenced depth_comple
            if 'depth_comple' in batch_img_metas[0].keys():
                depth_comple = [img_meta['depth_comple'] for img_meta in batch_img_metas]
                depth_comple = torch.tensor(np.stack(depth_comple, axis=0)).to(device).unsqueeze(1)
            else:
                depth_comple = torch.zeros_like(gt_depths).to(device)
            radar_depth = [img_meta['radar_depth'] for img_meta in batch_img_metas]
            radar_depth = torch.tensor(np.stack(radar_depth, axis=0)).to(device).unsqueeze(1)
            radar_depth = radar_depth.to(torch.float32)
            # preprocessed bbox_Mask and segmentation for msk2D supervison, NOTE: downsampled
            h, w = batch_img_metas[0]['img_shape']
            h_down, w_down = h // self.downsample, w // self.downsample
            if 'segmentation' in batch_img_metas[0].keys():
                segmentation = [img_meta['segmentation'].astype(np.float32) for img_meta in batch_img_metas]
                segmentation = torch.tensor(np.stack(segmentation, axis=0), dtype=torch.float32).to(device).unsqueeze(1)
                segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
            else:
                segmentation = torch.zeros((len(batch_img_metas), 1, h_down, w_down), dtype=torch.float32).to(device)
            bbox_Mask = [img_meta['bbox_Mask'] for img_meta in batch_img_metas]
            bbox_Mask = torch.tensor(np.stack(bbox_Mask, axis=0)).to(device).unsqueeze(1)
            bbox_Mask = F.interpolate(bbox_Mask, (h_down, w_down), mode='bilinear', align_corners=True)
        else:
            batch_img_metas = batch_img_metas[0]
            if 'segmentation' in batch_img_metas:
                orig_seg = batch_img_metas['segmentation']
            if 'radar_depth' in batch_img_metas:
                radar_depth_orig = batch_img_metas['radar_depth']
            if 'depth_comple' in batch_img_metas:
                depth_comple_orig = batch_img_metas['depth_comple']

            h, w = batch_img_metas['img_shape']
            if 'gt_bboxes_3d' in batch_img_metas:
                gt_bboxes_3d = [batch_img_metas['gt_bboxes_3d']]
            else:
                gt_bboxes_3d = []
            if 'gt_labels_3d' in batch_img_metas:
                gt_labels_3d = [batch_img_metas['gt_labels_3d']]
            else:
                gt_labels_3d = []
            if 'gt_bboxes_2d' in batch_img_metas:
                gt_bboxes_2d = [batch_img_metas['gt_bboxes_2d']]

            else:
                gt_bboxes_2d = []
            if 'gt_labels_2d' in batch_img_metas:
                gt_labels_2d = [batch_img_metas['gt_labels_2d']]
            else:
                gt_labels_2d = []
            if 'gt_depths' in batch_img_metas:
                gt_depths = [batch_img_metas['gt_depths']]
                gt_depths = torch.stack(gt_depths).unsqueeze(1)  # B, 1, H, W
                gt_depths = gt_depths.to(device)
            else:
                gt_depths = torch.zeros((1, 1, h, w)).to(device)
            H, W = batch_img_metas['img_shape']

            cam_aware = batch_img_metas['cam_aware']
            cam_aware = [[x.to(device)] for x in cam_aware]
            cam_aware = [torch.stack(x, dim=0) for x in cam_aware]
            img_aug_matrix = [batch_img_metas['img_aug_matrix']]
            img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
            img_aug_matrix = img_aug_matrix.to(device)
            if 'lidar_aug_matrix' in batch_img_metas:
                lidar_aug_matrix = [batch_img_metas['lidar_aug_matrix']]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32)
                bda_rot = [batch_img_metas['bda_rot']]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32)
            else:
                lidar_aug_matrix = torch.eye(4).unsqueeze(0)
                bda_rot = lidar_aug_matrix

            lidar_aug_matrix = lidar_aug_matrix.to(device)
            bda_rot = bda_rot.to(device)
            gt_bev_mask = torch.zeros((1, 1, self.bev_h_, self.bev_w_)).to(device)
            batch_img_metas = self.reorganize_lidar2img([batch_img_metas])  # begin list again
            calib = []
            mat = batch_img_metas[0]['final_lidar2img']
            mat = torch.Tensor(mat).to(device)
            final_lidar2img = torch.stack([mat])
            if 'depth_comple' in batch_img_metas[0].keys():
                depth_comple = [batch_img_metas[0]['depth_comple']] if isinstance(batch_img_metas, list) else [
                    batch_img_metas['depth_comple']]
                depth_comple = torch.tensor(np.stack(depth_comple, axis=0)).to(device).unsqueeze(1)

            else:
                depth_comple = torch.zeros((1, 1, H, W)).to(device)
            radar_depth = [batch_img_metas[0]['radar_depth']] if isinstance(batch_img_metas, list) else [
                batch_img_metas['depth_comple']]
            radar_depth = torch.tensor(np.stack(radar_depth, axis=0)).to(device).unsqueeze(1)
            radar_depth = radar_depth.to(torch.float32)
            h, w = batch_img_metas[0]['img_shape']
            h_down, w_down = h // self.downsample, w // self.downsample
            if 'segmentation' in batch_img_metas[0].keys():
                segmentation = [batch_img_metas[0]['segmentation'].astype(np.float32)] if isinstance(batch_img_metas,
                                                                                                     list) else [
                    batch_img_metas['segmentation']]
                segmentation = torch.tensor(np.stack(segmentation, axis=0)).to(device).unsqueeze(1)
                segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
            else:
                segmentation = torch.zeros((1, 1, h_down, w_down)).to(device)
            bbox_Mask = [batch_img_metas[0]['bbox_Mask']] if isinstance(batch_img_metas, list) else [
                batch_img_metas['bbox_Mask']]
            bbox_Mask = torch.tensor(np.stack(bbox_Mask, axis=0)).to(device).unsqueeze(1)
            bbox_Mask = F.interpolate(bbox_Mask, (h_down, w_down), mode='bilinear', align_corners=True)

        return batch_img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, \
            cam_aware, img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask, final_lidar2img

    def reorganize_lidar2img(self, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for img_metas in batch_input_metas:
            final_cam2img = copy.deepcopy(img_metas['cam2img'])
            final_lidar2img = copy.deepcopy(img_metas['lidar2img'])

            # same as visualization in BEVAug3D
            rots, trans, intrins, post_rots, post_trans = img_metas['cam_aware'][:5]
            final_cam2img[:2, :3] = post_rots[:2, :2] @ final_cam2img[:2, :3]
            final_cam2img[:2, 2] = post_trans[:2] + final_cam2img[:2, 2]
            final_lidar2img = final_cam2img @ img_metas['lidar2cam']
            final_lidar2img = final_lidar2img @ np.linalg.inv(img_metas['lidar_aug_matrix'])
            img_metas['final_lidar2img'] = final_lidar2img

        return batch_input_metas

    def generate_bev_mask(self, gt_bboxes_3d, batch_size, device, occ_threshold):
        # As long as it is occupied, it is 1
        gt_bev_mask = []
        if len(gt_bboxes_3d) != 0:
            bev_cell_size = torch.tensor(self.bev_cell_size).to(device)
            for bsid in range(len(gt_bboxes_3d)):
                bev_mask = torch.zeros(self.bev_grid_shape)
                current_boxes = gt_bboxes_3d[bsid]
                if len(current_boxes.tensor) == 0:
                    gt_bev_mask.append(bev_mask.to(torch.bool))
                    continue
                bbox_corners = gt_bboxes_3d[bsid].corners[:, [0, 2, 4, 6], :2]  # bev corners
                num_rectangles = bbox_corners.shape[0]
                bbox_corners[:, :, 0] = (bbox_corners[:, :, 0] - self.xbound[0]) / bev_cell_size[0]  # id_num, 4, 2
                bbox_corners[:, :, 1] = (bbox_corners[:, :, 1] - self.ybound[0]) / bev_cell_size[1]  # id_num, 4, 2

                # precise bur slow method
                grid_min = torch.clip(torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64), 0,
                                      self.bev_grid_shape[0] - 1)
                grid_max = torch.clip(torch.ceil(torch.max(bbox_corners, axis=1).values).to(torch.int64), 0,
                                      self.bev_grid_shape[1] - 1)
                possible_mask_h_all = torch.cat([grid_min[:, 0:1], grid_max[:, 0:1]], dim=1).tolist()
                possible_mask_w_all = torch.cat([grid_min[:, 1:2], grid_max[:, 1:2]], dim=1).tolist()
                for n in range(num_rectangles):
                    clock_corners = bbox_corners[n].cpu().numpy()[(0, 1, 3, 2), :]
                    poly = Polygon(clock_corners)
                    h_list = possible_mask_h_all[n];
                    h_list = np.arange(h_list[0] - 1, h_list[1] + 1, 1);
                    h_list = np.clip(h_list, 0, self.bev_grid_shape[0] - 1)
                    w_list = possible_mask_w_all[n];
                    w_list = np.arange(w_list[0] - 1, w_list[1] + 1, 1);
                    w_list = np.clip(w_list, 0, self.bev_grid_shape[1] - 1)
                    for i in h_list:
                        for j in w_list:
                            cell_center = np.array([i + 0.5, j + 0.5])
                            cell_poly = box(i, j, i + 1, j + 1)
                            if poly.contains(Point(cell_center)):
                                bev_mask[i, j] = True
                            else:
                                intersection = cell_poly.intersection(poly)
                                if (intersection.area / cell_poly.area) > occ_threshold: bev_mask[i, j] = True
                # coarse but quick method
                # for i in range(num_rectangles):
                #     bev_mask[grid_min[i, 0]:grid_max[i, 0], grid_min[i, 1]:grid_max[i, 1]] = True
                # save_image(bev_mask[None,None,:,:]*0.99, 'gt_bev_mask.png')
                gt_bev_mask.append(bev_mask)
            gt_bev_mask = torch.stack(gt_bev_mask, dim=0).unsqueeze(1)  # B 1 H W
        else:
            gt_bev_mask = torch.zeros((batch_size, 1, self.bev_grid_shape[0], self.bev_grid_shape[1]))
        gt_bev_mask = gt_bev_mask.to(torch.bool)
        return gt_bev_mask

    def recording_fps(self, step_all_time):
        self.record_fps['num'] += 1
        self.record_fps['time'] += step_all_time
        if not self.training and self.record_fps['num'] % 50 == 0:
            print(' FPS: %.2f' % (1.0 / step_all_time))
        if not self.training and self.record_fps['num'] == 1296 and not self.training:
            print(' FINAL VOD FPS: %.2f' % (1296 / self.record_fps['time']))

    @master_only
    def draw_gt_pred_figures_3d(self, points, imgs, gt_bboxes_3ds, gt_labels_3ds, img_metas, rescale=False,
                                threshold=0.3,
                                bbox_list=None, override_save_dir=None, **kwargs):
        # if training we should decode the bbox from features 'outs_pts' first
        self.vis_time_box3d += 1
        if not self.vis_time_box3d % self.SAVE_INTERVALS == 0: return

        if override_save_dir:
            figures_path_det3d = override_save_dir
        elif self.training:
            figures_path_det3d = self.figures_path_det3d_train
        else:
            figures_path_det3d = self.figures_path_det3d_test
        os.makedirs(figures_path_det3d, exist_ok=True)

        device = next(self.parameters()).device
        gt_bboxes_3ds = [b.to(device) for b in gt_bboxes_3ds]
        gt_labels_3ds = [l.to(device) for l in gt_labels_3ds]

        # filter out the ignored labels
        gt_bboxes_3ds_filtered = []
        for i in range(len(img_metas)):
            index_mask = (gt_labels_3ds[i] != -1)
            gt_bboxes_3ds_filtered.append(gt_bboxes_3ds[i][index_mask])
        gt_bboxes_3ds = gt_bboxes_3ds_filtered

        outs_pts = kwargs.get('outs_pts')
        if bbox_list is None and outs_pts is not None:
            bbox_results = self.pts_bbox_head.get_bboxes(*outs_pts, img_metas, rescale=rescale)
            bbox_list = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_results]
        elif bbox_list is None:
            bbox_list = [None] * len(img_metas)

        # starting visualization
        for i in range(imgs.shape[0]):  # batch size
            # preparation
            input_img = np.array(imgs[i].cpu()).transpose(1, 2, 0)
            input_img = input_img * self.std[None, None, :] + self.mean[None, None, :]
            current_bbox_results = bbox_list[i]
            if current_bbox_results and 'boxes_3d' in current_bbox_results:
                pred_bboxes_3d = current_bbox_results['boxes_3d']
                pred_scores_3d = current_bbox_results['scores_3d']
                pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d > threshold].to('cpu')
                if len(pred_bboxes_3d) == 0:
                    pred_bboxes_3d = None
            else:
                pred_bboxes_3d = None

            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            proj_mat = img_metas[i]["final_lidar2img"]
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]

            # draw in image view
            filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'

            # draw in bev view
            save_path = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev.png')
            save_path_paper = os.path.join(figures_path_det3d,
                                           str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper.png')
            point = points[i].cpu().detach().numpy()[:, :3]

            pd_bbox_corners = pred_bboxes_3d.corners[:, [0, 2, 4, 6], :2].numpy()[:, (0, 1, 3, 2),
                              :] if pred_bboxes_3d is not None else None
            gt_bbox_corners = gt_bboxes_3d.corners[:, [0, 2, 4, 6], :2].numpy()[:, (0, 1, 3, 2),
                              :] if gt_bboxes_3d is not None else None

            draw_bev_pts_bboxes(point, gt_bbox_corners, pd_bbox_corners, save_path=save_path, xlim=self.xlim,
                                ylim=self.ylim)

            # for paper figures
            tmp_img_true = custom_draw_lidar_bbox3d_on_img(gt_bboxes_3d, input_img.copy(), proj_mat, img_metas,
                                                           color=(61, 102, 255), thickness=2, scale_factor=3)
            tmp_img_pred = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, input_img.copy(), proj_mat, img_metas,
                                                           color=(241, 101, 72), thickness=2, scale_factor=3)
            tmp_img_alls = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, tmp_img_true.copy(), proj_mat, img_metas,
                                                           color=(241, 101, 72), thickness=2, scale_factor=3)
            mmcv.imwrite(tmp_img_true, os.path.join(figures_path_det3d, f'{filename}_gt.png'))
            mmcv.imwrite(tmp_img_pred, os.path.join(figures_path_det3d, f'{filename}_pred.png'))
            mmcv.imwrite(tmp_img_alls, os.path.join(figures_path_det3d, f'{filename}.png'))
            draw_paper_bboxes(point, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper, xlim=self.xlim,
                              ylim=self.ylim)

    @master_only
    def draw_gt_pred_bev(self, gt_bev_mask, bev_mask, bev_mask_logit_sigmoid, img_metas, mask_thre, suffix='former'):
        if suffix == 'former': self.vis_time_bev2d += 1
        if not self.vis_time_bev2d % self.SAVE_INTERVALS == 0: return
        if self.training:
            figures_path_bev2d = self.figures_path_bev2d_train
        else:
            figures_path_bev2d = self.figures_path_bev2d_test

        bev1 = torch.rot90(gt_bev_mask, k=1, dims=(2, 3))
        bev2 = torch.rot90(bev_mask, k=1, dims=(2, 3))
        bev3 = torch.rot90(bev_mask_logit_sigmoid, k=1, dims=(2, 3))
        bev4 = torch.rot90(bev_mask_logit_sigmoid > mask_thre, k=1, dims=(2, 3))
        b, _, h, w = bev1.shape
        frame_1 = 0.5 * torch.ones((1, h, 5)).to(bev_mask_logit_sigmoid.device)
        for i in range(bev_mask.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            save_bev = torch.cat([frame_1, bev1[i], frame_1, bev2[i], frame_1, bev3[i], frame_1, bev4[i], frame_1],
                                 dim=2) * 0.99
            frame_2 = 0.5 * torch.ones((1, 5, save_bev.shape[2])).to(bev_mask_logit_sigmoid.device)
            save_bev = torch.cat([frame_2, save_bev, frame_2], dim=1)
            save_image(save_bev, os.path.join(figures_path_bev2d,
                                              str(self.vis_time_bev2d) + '_' + img_name + '_bev2d_' + suffix + '.png'))

    @master_only
    def draw_bev_feature_map(self, bev_feats, img_metas, bev_feats_name='bev_feats_fusion'):
        # if bev_feats_name=='bev_feats_fusion_refined': self.vis_time_bevnd += 1
        # if not self.vis_time_bevnd % self.SAVE_INTERVALS == 0: return
        if self.training:
            figures_path_bevnd = self.figures_path_bevnd_train
        else:
            figures_path_bevnd = self.figures_path_bevnd_test

        b, _, h, w = bev_feats.shape
        # bev_feats = bev_feats.mean(1).unsqueeze(1) # using mean
        bev_feats_show = bev_feats.max(1, keepdim=True).values  # using max
        # bev_feats_show = torch.rot90(bev_feats_show, k=2, dims=(2, 3))\
        bev_feats_show = torch.flip(bev_feats_show, [2])  # horizontal flip for consistency to gt bev bbox
        for i in range(bev_feats.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            bev_feats_tmp = bev_feats_show[i:i + 1, :, :, :]
            bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min()) / (bev_feats_tmp.max() - bev_feats_tmp.min())
            # bev_feats_tmp = (bev_feats_tmp - 0.75)/(1.00 - 0.75)
            if bev_feats_name == 'bev_feats_img': bev_feats_tmp = bev_feats_tmp * 25
            bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
            bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3]
            bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
            save_image(bev_feats_tmp_colored, os.path.join(figures_path_bevnd,
                                                           str(self.vis_time_bevnd) + '_' + img_name + '_' + bev_feats_name + '.png'))
            if not os.path.exists(figures_path_bevnd):
                # print(f"WARNING: Directory {figures_path_bevnd} does not exist. Creating it...")
                os.makedirs(figures_path_bevnd, exist_ok=True)

    @master_only
    def draw_gt_pred_rangeview(self, img_metas, segs, gts, preds, sigmoids, eroded=False):
        if not eroded: self.vis_time_range += 1
        if not self.vis_time_range % self.SAVE_INTERVALS == 0: return
        if self.training:
            figures_path_range = self.figures_path_range_train
        else:
            figures_path_range = self.figures_path_range_test

        b, _, h, w = gts.shape
        frame_1 = 0.5 * torch.ones((1, 5, w)).to(preds.device)
        for i in range(gts.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            seg = segs[i];
            gt = gts[i];
            pred = preds[i];
            sigmoid = sigmoids[i]
            save_range = torch.cat([frame_1, seg, frame_1, gt, frame_1, pred, frame_1, sigmoid, frame_1], dim=1)
            frame_2 = 0.5 * torch.ones((1, save_range.shape[1], 5)).to(pred.device)
            save_range = torch.cat([frame_2, save_range, frame_2], dim=2)
            if not eroded:
                save_image(save_range,
                           os.path.join(figures_path_range, str(self.vis_time_range) + '_' + img_name + '_range.png'))
            else:
                save_image(save_range, os.path.join(figures_path_range,
                                                    str(self.vis_time_range) + '_' + img_name + '_range_eroded.png'))

    @master_only
    def draw_pts_completion(self, img_metas, gt_points, pd_points, gt_bboxes_3d=None, gt_labels_3d=None,
                            plot_mode='distance', points_type='pointpainting'):
        if points_type == 'voxelpainting': self.vis_time_point += 1
        if not self.vis_time_point % self.SAVE_INTERVALS == 0: return
        if self.training:
            figures_path_point = self.figures_path_point_train
        else:
            figures_path_point = self.figures_path_point_test

        for i in range(len(img_metas)):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            for ax, points, title in zip(axes, [gt_points, pd_points], ['Raw Points', 'Virtual Points']):
                points = points[i][(self.xlim[0] <= points[i][:, 0]) & (points[i][:, 0] <= self.xlim[1]) & \
                                   (self.ylim[0] <= points[i][:, 1]) & (points[i][:, 1] <= self.ylim[1])]
                ax.set_xlim(self.xlim[0], self.xlim[1])
                ax.set_ylim(self.ylim[0], self.ylim[1])
                ax.autoscale(False)

                # plot points
                points = points.cpu().detach().numpy()
                x = points[:, 0]
                y = points[:, 1]
                if plot_mode == 'distance':
                    intensities = np.clip(np.sqrt(x ** 2 + y ** 2) / 60, 0, 1)
                    colors = plt.cm.gray(intensities)
                if plot_mode == 'RCS':
                    norm_max = np.max(gt_points[i].cpu().detach().numpy()[:, 3])
                    norm_min = np.min(gt_points[i].cpu().detach().numpy()[:, 3])
                    intensities = np.clip((points[:, 3] - norm_min) / (norm_max - norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'v_r_compensated':
                    norm_max = np.max(gt_points[i].cpu().detach().numpy()[:, 4])
                    norm_min = np.min(gt_points[i].cpu().detach().numpy()[:, 4])
                    intensities = np.clip((points[:, 4] - norm_min) / (norm_max - norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'logits':
                    norm_max = np.max(points[:, -1])
                    norm_min = np.min(points[:, -1])
                    intensities = np.clip((points[:, -1] - norm_min) / (norm_max - norm_min), 0, 1)
                    intensities = 1 - intensities
                    intensities = intensities * 0.6 + 0.2  # 0.2 - 0.8
                    colors = plt.cm.gray(intensities)
                if plot_mode == 'context_pointpainting':
                    context = np.sum(points[:, 5:], axis=-1)
                    norm_max = np.max(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    norm_min = np.min(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    intensities = np.clip((context - norm_min) / (norm_max - norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'context_voxelpainting':
                    context = np.sum(points[:, 5:], axis=-1)
                    norm_max = np.max(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    norm_min = np.min(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    intensities = np.clip((context - norm_min) / (norm_max - norm_min), 0, 1)
                    intensities = 1 - intensities
                    intensities = intensities * 0.6 + 0.2  # 0.2 - 0.8
                    colors = plt.cm.gray(intensities)
                    sorted_indices = np.argsort(-intensities)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    colors = colors[sorted_indices]
                ax.scatter(x, y, c=colors, s=15)  # alpha=0.5

                # plot bboxes
                if gt_bboxes_3d is not None:
                    if len(gt_bboxes_3d) != 0:
                        gt_bboxes_3d_filtered = gt_bboxes_3d[i][gt_labels_3d[i] != -1]
                        gt_bbox_corners = gt_bboxes_3d_filtered.corners[:, [0, 2, 4, 6], :2]
                        gt_bbox_corners = gt_bbox_corners.cpu().detach().numpy()[:, (0, 1, 3, 2), :]  # clock_corners
                        for bbox in gt_bbox_corners:
                            polygon = patches.Polygon(bbox, closed=True, edgecolor='red', linewidth=1, fill=False)
                            ax.add_patch(polygon)

                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'Point cloud and bboxes under BEV - {title}')
                ax.grid(True)

            save_path = os.path.join(figures_path_point,
                                     str(self.vis_time_point) + '_' + img_name + '_' + points_type + '.png')
            plt.savefig(save_path)
            plt.close(fig)