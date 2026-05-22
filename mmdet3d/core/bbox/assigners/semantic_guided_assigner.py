import torch
from mmdet.core.bbox.assigners import MaxIoUAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS


@BBOX_ASSIGNERS.register_module()
class SemanticGuidedAssigner(MaxIoUAssigner):
    """
    根据MaxIoU和来自2D proposal的BEV语义掩码，为Anchor分配GT。
    """

    def __init__(self,
                 point_cloud_range,
                 grid_size,
                 promote_to_positive_thr,
                 **kwargs):
        super().__init__(**kwargs)


        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.grid_size = torch.tensor(grid_size, dtype=torch.int32)
        self.promote_to_positive_thr = promote_to_positive_thr

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, bev_semantic_mask=None, **kwargs):
        # 1. 得到基准分配结果
        assign_result = super().assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)


        if bev_semantic_mask is None or gt_labels is None or len(gt_labels) == 0:
            return assign_result


        anchor_centers = bboxes.gravity_center[:, :2].to(bev_semantic_mask.device)  # xy

        # LiDAR坐标转换为BEV坐标
        pc_range = self.point_cloud_range.to(anchor_centers.device)
        grid_size = self.grid_size.to(anchor_centers.device)

        scale_x = grid_size[0] / (pc_range[3] - pc_range[0])
        scale_y = grid_size[1] / (pc_range[4] - pc_range[1])

        grid_x = ((anchor_centers[:, 0] - pc_range[0]) * scale_x).long()
        grid_y = ((anchor_centers[:, 1] - pc_range[1]) * scale_y).long()
        grid_x = torch.clamp(grid_x, 0, grid_size[0] - 1)
        grid_y = torch.clamp(grid_y, 0, grid_size[1] - 1)


        is_in_semantic_region = bev_semantic_mask[grid_y, grid_x] > 0.5

        assigned_gt_inds = assign_result.gt_inds

        potential_promotes_mask = (assigned_gt_inds == 0) & is_in_semantic_region

        if not potential_promotes_mask.any():
            return assign_result

        potential_anchors = bboxes[potential_promotes_mask]

        iou_with_gt = self.iou_calculator(potential_anchors, gt_bboxes)

        max_iou, max_gt_inds = iou_with_gt.max(dim=1)

        promote_mask = max_iou > self.promote_to_positive_thr

        if not promote_mask.any():
            return assign_result

        # 获取最终要被提拔的Anchor在原始列表中的索引
        promote_indices = torch.where(potential_promotes_mask)[0][promote_mask]

        assign_result.gt_inds[promote_indices] = max_gt_inds[promote_mask] + 1
        assign_result.labels[promote_indices] = gt_labels[max_gt_inds[promote_mask]]

        return assign_result