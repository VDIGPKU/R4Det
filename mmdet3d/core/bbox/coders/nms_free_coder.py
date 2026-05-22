import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
import torch.nn.functional as F

from ..utils import denormalize_bbox, denormalize_bbox_polar


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """
    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 code_size=None):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.code_size =code_size

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='trunc')
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            limit = torch.tensor(self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >= limit[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= limit[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!'
            )

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))

        return predictions_list

@BBOX_CODERS.register_module()
class NMSFreeCoder1(BaseBBoxCoder):
    """
    适用于 DETR-like 3D 检测器的 BBox Coder (NMS-Free 风格后处理)。
    解码逻辑基于: 绝对坐标 = 反归一化(归一化参考点) + 相对偏移量。
    尺寸被假定为直接预测绝对值。
    """
    def __init__(self,
                 pc_range,
                 voxel_size=None, # 通常在解码中不直接使用
                 post_center_range=None,
                 max_num=100,      # TopK 选择的最大数量
                 score_threshold=None,
                 num_classes=10,
                 code_size=None):
        self.code_size = code_size
        self.pc_range = pc_range
        self.pc_range_torch = torch.tensor(pc_range)
        if len(self.pc_range_torch) == 6:
            self.pc_min = self.pc_range_torch[[0, 1, 2]]
            self.pc_max = self.pc_range_torch[[3, 4, 5]]
            self.pc_range_dims = self.pc_max - self.pc_min
        else:
            raise ValueError("pc_range must contain 6 elements [x_min, y_min, z_min, x_max, y_max, z_max]")

        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        if self.post_center_range is not None:
             self.post_center_range_torch = torch.tensor(post_center_range)
        else:
             self.post_center_range_torch = None
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes


    def encode(self, gt_bboxes, gt_labels=None):
        pass

    def decode(self, all_cls_scores, all_bbox_deltas, all_reference_points, img_metas):
        """为整个批次解码 Bounding Box。
        Args:
            all_cls_scores (Tensor): 最后一层的分类分数 logits。(B, N_q, NumClasses)
            all_bbox_deltas (Tensor): 最后一层的相对偏移/编码值。(B, N_q, CodeSize)
            all_reference_points (Tensor): 最后一层的归一化参考点。(B, N_q, 3)
            img_metas (list[dict]): 批次中每个样本的元信息。

        Returns:
            list[dict]: 批次中每个样本的解码结果列表。
                每个字典包含:
                    'bboxes': 检测框 (N_kept, CodeSize)，绝对坐标 (xc, yc, zc, w, l, h, sin, cos)。
                    'scores': 置信度分数 (N_kept,)。
                    'labels': 类别标签索引 (N_kept,)。
        """
        batch_size = all_cls_scores.size(0)
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_deltas[i], all_reference_points[i], img_metas[i])
            )
        return predictions_list

    def decode_single(self, cls_scores, bbox_deltas, reference_points, img_meta):
        """为单个样本解码 Bounding Box。
        Args:
            cls_scores (Tensor): 分类分数 logits。(N_q, NumClasses)
            bbox_deltas (Tensor): 相对偏移/编码值。(N_q, CodeSize)
            reference_points (Tensor): 归一化参考点。(N_q, 3)
            img_meta (dict): 当前样本的元信息 (可选，未来可能用于特定解码)。
        Returns:
            dict: 包含 'bboxes', 'scores', 'labels' 的字典。
        """
        max_num_per_sample = self.max_num # TopK 选择数量
        code_size = bbox_deltas.shape[-1] # 从输入推断 code_size (e.g., 8)

        # 确保所有计算在同一设备上
        device = cls_scores.device
        self.pc_min = self.pc_min.to(device)
        self.pc_range_dims = self.pc_range_dims.to(device)
        if self.post_center_range_torch is not None:
             self.post_center_range_torch = self.post_center_range_torch.to(device)



        cls_scores = cls_scores.sigmoid()
        scores, topk_indices = cls_scores.view(-1).topk(max_num_per_sample)
        labels = topk_indices % self.num_classes
        query_indices = torch.div(topk_indices, self.num_classes, rounding_mode='trunc')

        # 根据 Query 索引选出对应的偏移量和参考点
        selected_bbox_deltas = bbox_deltas[query_indices]      # (TopK, CodeSize)
        selected_reference_points = reference_points[query_indices] # (TopK, 3) - Normalized


        # 反归一化参考点 -> 绝对坐标
        ref_pts_abs = selected_reference_points * self.pc_range_dims + self.pc_min # (TopK, 3)

        decoded_centers_abs = ref_pts_abs + selected_bbox_deltas[:, :3]  # (TopK, 3)

        # 2. 解码尺寸 (添加 torch.exp，与 loss_single 一致)
        decoded_dims_abs = torch.exp(selected_bbox_deltas[:, 3:6])  # (TopK, 3)

        #decoded_sincos_abs =selected_bbox_deltas[:, 6:8]  # (TopK, 2)
        decoded_sincos_abs = F.normalize(selected_bbox_deltas[:, 6:8], p=2, dim=-1)
        # 拼接得到最终的绝对坐标预测
        # (TopK, CodeSize) , (TopK, 8) for (xc, yc, zc, w, l, h, sin, cos)
        final_box_preds = torch.cat([
            decoded_centers_abs,
            decoded_dims_abs,
            decoded_sincos_abs
        ], dim=1)

        final_scores = scores
        final_labels = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        else:
            thresh_mask = torch.ones_like(final_scores, dtype=torch.bool) #不过滤

        # 应用中心点范围过滤
        if self.post_center_range_torch is not None:
            limit = self.post_center_range_torch
            center_mask = (final_box_preds[:, :3] >= limit[:3]).all(1)
            center_mask &= (final_box_preds[:, :3] <= limit[3:]).all(1)

            mask = thresh_mask & center_mask
        else:
            mask = thresh_mask

        # 应用 mask 过滤
        boxes3d = final_box_preds[mask] # (N_kept, CodeSize) 绝对坐标
        scores = final_scores[mask]     # (N_kept,)
        labels = final_labels[mask]     # (N_kept,)

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels,
        }

        return predictions_dict


@BBOX_CODERS.register_module()
class NMSFreeCoderPolar(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox_polar(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores >= self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list