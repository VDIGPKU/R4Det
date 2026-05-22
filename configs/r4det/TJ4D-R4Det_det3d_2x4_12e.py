# initialization
custom_imports = dict(imports=['mmdet3d'])
find_unused_parameters = True
# dataset settings
dataset_type = 'TJ4DDataset'
data_root = 'data/TJ4D/'
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

# dataset BEV grid and pc range configs
point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
post_center_range = [x + y for x, y in zip(point_cloud_range, [-10, -10, -5, 10, 10, 5])]
voxel_size = [0.32, 0.32, 6.00] # [0.32, 0.32, 6.00] # [0.432, 0.496, 6.000]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_size[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_size[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
    'dbound': [1.0, 73, 1.0]}
code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
code_size = len(code_weights)

# ablation settings
backward_ablation = dict(pv_logits=True, depth_prob=True, cross=3, self=2)
focusradardepth_ablation = dict(foreground_loss_alpha=5.0)
painting_ablation = dict(pv_logits=False, depth_prob=True)

# supervision settings
use_box3d_supervision = True                # 3D object detection task
use_depth_supervision = True                # leverage depth estimation from lidar supervision for better view transformation
use_props_supervision = True                # BEV proposal network for better multi-modality fusion
use_msk2d_supervision = True                # default both 0.10 for pretrain, weak supervision from 2 GTs
# network structures
use_backward_projection = True              # LACA: localization-aware attention backward projection
use_sa_radarnet = True                      # SRP: whether use radar associate with image features
use_radar_depth = True                      # GDC: whether to project radar to image for better monodepth estimation
# default settings
assert use_box3d_supervision == True        # det3d pretraining, not training
freeze_depths = False                       # of course False because we are pretraining this module
freeze_radars = False                       # radars must be pretrained, load_radar_from is not None, False for aug train here
use_grid_mask = False                       # before pre-extract feats of raw-img
freeze_images = True                        # for baseline(BEVFusion) True, img_backbone and img_neck, dark img may need re-train
camera_stream = 'LSS'                       # camera stream lift method, exactly view transformation
loss_depth_prob = 0.1                       # default 0.10 for both pretrain and train
loss_bev_seg    = 0.1                       # default 0.10 for both pretrain and train
loss_range_seg  = 0.1                       # default 0.10 for both pretrain and train
use_inst_supervision = True
# image augumentation
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0], to_rgb=False,
)
ida_aug_conf = {
    'resize_lim': (0.40, 0.50),
    'final_dim': (480, 640),
    'final_dim_test': (480, 640), # 960 1280
    'bot_pct_lim': (0.0, 0.0),
    'top_pct_lim': (0.0, 0.1),
    'rot_lim': (-2.7, 2.7),
    'rand_flip': True,
}
# BEVDataAugmentation
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(1.0, 1.0, 0.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5,
)
inst_feat_dim = 256
# model parameter
bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
img_channels = 256
rad_channels = 384
downsample = 8
num_in_height = 8
_num_layers_cross_ = backward_ablation['cross']
_num_points_pillr_ = 8
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
num_3d_proposals = 300

_num_layers_self_ = backward_ablation['self']
_num_points_self_ = 8

bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
bev_grid_size = [bev_h_, bev_w_, 1] #  [216, 248, 1]

#code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#code_size = 7
#out_size_factor = 1
# Calculate grid size for CenterPoint head
#grid_size = [int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
#             int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
#             1]

# model settings
model = dict(
    type='R4Det',
    mask_save_dir='./work_dirs/my_mask_outputs/',
    bev_h_=bev_h_,
    bev_w_=bev_w_,
    img_channels=img_channels,
    rad_channels=rad_channels,
    num_classes=len(class_names),
    num_in_height=num_in_height,
    use_depth_supervision=use_depth_supervision,
    use_props_supervision=use_props_supervision,
    use_box3d_supervision=use_box3d_supervision,
    use_backward_projection=use_backward_projection,
    use_sa_radarnet=use_sa_radarnet,
    use_grid_mask=use_grid_mask,
    freeze_images=freeze_images,
    freeze_depths=freeze_depths,
    freeze_radars=freeze_radars,
    camera_stream=camera_stream,
    point_cloud_range=point_cloud_range,
    grid_config=grid_config,
    img_norm_cfg=img_norm_cfg,
    backward_ablation=backward_ablation,
    focusradardepth_ablation=focusradardepth_ablation,
    painting_ablation=painting_ablation,

    img_rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    img_roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=4,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    instance_feature_fusion_layer=dict(in_channels=_dim_ + inst_feat_dim, out_channels=_dim_),

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # TODO: figure if requires_grad
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=img_channels,
        # make the image features more stable numerically to avoid loss nan
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    img_view_transformer=dict(
        type='ViewTransformerLSS',
        downsample=downsample,
        grid_config=grid_config,
        data_config=ida_aug_conf),
    depth_net=dict(
        type='GeometryDepth_Net',
        use_radar_depth=use_radar_depth,
        use_extra_depth=False,
        data_config=ida_aug_conf,
        downsample=downsample,
        input_channels=img_channels * 5,
        numC_input=_dim_,
        numC_Trans=_dim_,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_prob_weight=loss_depth_prob,
        loss_abs_weight=0.01,
        loss_sam2_weight=0.02,
        foreground_loss_alpha=focusradardepth_ablation['foreground_loss_alpha'],
        relative_loss_weight=0.04),
    rangeview_foreground=dict(
        type='MRF3Net',
        input_channel=_dim_,
        output_channel=1,
        base_channel=_dim_,
        mask_thre_train=0.95,
        mask_thre_test=0.70,
        loss_box=loss_range_seg,
        loss_seg=loss_range_seg),
    pts_voxel_layer=dict(
        max_num_points=10, # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
        max_voxels=(16000, 40000)),  # (training, testing) max_voxels
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
        point_cloud_range=point_cloud_range,
        legacy=False,
        with_velocity_snr_center=True),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[bev_w_*2, bev_h_*2]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    RCFusion=dict(
        type='Cross_Modal_Fusion',
        kernel_size=3,
        img_channels=_dim_,
        rad_channels=rad_channels,
        out_channels=_dim_),
    proposal_layer=dict(
        type='FRPN',
        in_channels=_dim_,
        scale_factor=1.0,
        mask_thre=0.4,
        topk_rate_test=0.01,
        loss_weight=loss_bev_seg),
    temporal_fusion=dict(
        type='TemporalDeformableFusion',
        in_channels=_dim_,
        out_channels=_dim_,
        kernel_size=3,
        deform_groups=1,
        gate_kernel_size=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True)
    ),
    backward_projection=dict(
        type='BackwardProjection',
        bev_h_=bev_h_,
        bev_w_=bev_w_,
        grid_config=grid_config,
        data_config=ida_aug_conf,
        in_channels=_dim_,
        out_channels=_dim_,
        point_cloud_range=point_cloud_range,
        mlp_prior=True,
        embed_dims = _dim_,
        cross_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams = _num_cams_,
            encoder=dict(
                type='BEVFormerEncoder_DFA3D',
                num_layers=_num_layers_cross_,
                pc_range=point_cloud_range,
                data_config=ida_aug_conf,
                num_points_in_pillar=_num_points_pillr_,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention_DFA3D',
                            pc_range=point_cloud_range,
                            num_cams=_num_cams_,
                            deformable_attention=dict(
                               type='MSDeformableAttention3D_DFA3D',
                               embed_dims=_dim_,
                               num_points=_num_points_cross_,
                               num_levels=_num_levels_),
                            embed_dims=_dim_,
                       )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams = _num_cams_,
            use_level_embeds = False,
            use_cams_embeds = False,
            encoder=dict(
                type='BEVFormerEncoder_DFA3D',
                num_layers=_num_layers_self_,
                pc_range=point_cloud_range,
                data_config=ida_aug_conf,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_points=_num_points_self_)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,)),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(class_names),
        in_channels=_dim_,
        feat_channels=_dim_,
        use_direction_classifier=True,
        dir_offset=0.0,
        dir_limit_offset=1.0,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -1.163, 70.4, 40.0, -1.163],
                [0, -40.0, -1.353, 70.4, 40.0, -1.353],
                [0, -40.0, -1.363, 70.4, 40.0, -1.363],
                [0, -40.0, -1.403, 70.4, 40.0, -1.403],
            ],
            sizes=[[0.6, 0.8, 1.69], [0.78, 1.77, 1.60], [1.84, 4.56, 1.70],[2.66, 10.76, 3.47]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg = dict(
       img_rpn=dict(
           assigner=dict(
               type='MaxIoUAssigner',
               pos_iou_thr=0.7,
               neg_iou_thr=0.3,
               min_pos_iou=0.3,
               match_low_quality=True,
               ignore_iof_thr=-1),
           sampler=dict(
               type='RandomSampler',
               num=256,
               pos_fraction=0.5,
               neg_pos_ub=-1,
               add_gt_as_proposals=False),
           allowed_border=-1,
           pos_weight=-1,
           debug=False),
       img_rpn_proposal=dict(
           nms_across_levels=False,
           nms_pre=2000,
           nms_post=1000,
           max_per_img=1000,
           nms=dict(type='nms', iou_threshold=0.7),
           min_bbox_size=0),
       img_rcnn=dict(
           assigner=dict(
               type='MaxIoUAssigner',
               pos_iou_thr=0.5,
               neg_iou_thr=0.5,
               min_pos_iou=0.5,
               match_low_quality=True,
               ignore_iof_thr=-1),
           sampler=dict(
               type='RandomSampler',
               num=512,
               pos_fraction=0.25,
               neg_pos_ub=-1,
               add_gt_as_proposals=True),
           mask_size=28,
           pos_weight=-1,
           debug=False),
       pts=dict(
           assigner=[
               dict(  # for Pedestrian
                   type='MaxIoUAssigner',
                   iou_calculator=dict(type='BboxOverlapsNearest3D'),
                   pos_iou_thr=0.35,
                   neg_iou_thr=0.2,
                   min_pos_iou=0.2,
                   ignore_iof_thr=-1),
               dict(  # for Cyclist
                   type='MaxIoUAssigner',
                   iou_calculator=dict(type='BboxOverlapsNearest3D'),
                   pos_iou_thr=0.35,
                   neg_iou_thr=0.2,
                   min_pos_iou=0.2,
                   ignore_iof_thr=-1),
               dict(  # for Car
                   type='MaxIoUAssigner',
                   iou_calculator=dict(type='BboxOverlapsNearest3D'),
                   pos_iou_thr=0.5,
                   neg_iou_thr=0.35,
                   min_pos_iou=0.35,
                   ignore_iof_thr=-1),
               dict(  # for Truck
                   type='MaxIoUAssigner',
                   iou_calculator=dict(type='BboxOverlapsNearest3D'),
                   pos_iou_thr=0.5,
                   neg_iou_thr=0.35,
                   min_pos_iou=0.35,
                   ignore_iof_thr=-1),
           ],
           allowed_border=0,
           pos_weight=-1,
           debug=False)),
    test_cfg=dict(
        img_rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        img_rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5),
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=num_3d_proposals,
            max_num=num_3d_proposals,
            score_thr=0.0,
            nms_thr=0.5,
            min_bbox_size=0
        )
        )
)

# pipline settings
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5]),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='loadSegmentation', data_root=data_root, dataset='TJ4D', seg_type='detectron2'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='LoadVLSAMAnnotations'),
    dict(type='LoadMyDepthFromFile',  depth_base_path='/data/tangyousen/TJ4DRadSet_4DRadar/training/depth_npy_predict'),
    dict(type='ImageAug3D2', data_aug_conf=ida_aug_conf, is_train=True),
    dict(type='DownsampleDepthMap', target_shape=(60, 80)),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='TJ4D'),
    dict(type='CreateDepthFromRaDAR', filter_min=0.0, filter_max=80.0),
    dict(type='gen2DMask', use_seg=False, use_softlabel=False, is_train=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D2', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'gt_masks', 'my_gt_depth'],#,'bev_semantic_mask'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_aug_matrix', 'lidar_aug_matrix',
                    'lidar2cam', 'gt_depths', 'cam_aware', 'bda_rot',
                    'segmentation', 'depth_comple', 'bbox_Mask', 'radar_depth')),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5]),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='loadSegmentation', data_root=data_root, dataset='TJ4D', seg_type='detectron2'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='ImageAug3D2', data_aug_conf=ida_aug_conf, is_train=False),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='CreateDepthFromRaDAR', filter_min=0.0, filter_max=80.0),
    dict(type='gen2DMask', use_seg=False, use_softlabel=False, is_train=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D2', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_aug_matrix', 'lidar_aug_matrix',
                    'lidar2cam', 'gt_depths', 'cam_aware', 'bda_rot',
                    'segmentation', 'depth_comple', 'bbox_Mask', 'radar_depth')),
]
eval_pipeline = test_pipeline

# dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'TJ4D_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'TJ4D_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'TJ4D_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='LiDAR'))

# Training settings
lr = 0.0002
max_epochs = 18
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01,
                 paramwise_cfg=dict(
                     custom_keys={
                         'ifgd_fusion.conv_gamma': dict(lr_mult=0.1),
                         'ifgd_fusion.conv_beta': dict(lr_mult=0.1),
                         'ifgd_fusion.ebev_bn': dict(lr_mult=0.1),
                     }
                 ))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
momentum_config = None

# log checkpoint & evaluation
evaluation = dict(interval=1, pipeline=eval_pipeline)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'

# You may need to download the model first is the network is unstable
load_from ='work_dirs/TJ4D-R4Det_det3d_2x4_12e-final/epoch_2.pth'
resume_from = None
workflow = [('train', 1)]
device = 'cuda'