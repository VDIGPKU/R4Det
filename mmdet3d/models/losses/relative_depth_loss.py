import torch
import torch.nn as nn
from .utils import weight_reduce_loss
from ..builder import LOSSES
from ..sparsebev.sparsebev_sampling import sampling_4d
import torch.nn.functional as F


@LOSSES.register_module()
class SigmoidRankingLoss(nn.Module):
    """
    Sigmoid-based pairwise ranking loss with two-branch sampling:
    1. Foreground-biased pairs (from weight_map > 0)
    2. Global random pairs (whole image)
    """

    def __init__(self,
                 reduction='mean'):
        super(SigmoidRankingLoss, self).__init__()
        self.reduction = reduction

    def compute_pairwise_loss(self, pred_flat, gt_flat, weight_flat, mask_flat, num_samples):
        """
        Compute weighted sigmoid ranking loss for given mask.
        """
        total_weighted_loss = 0.0
        total_weight = 0.0
        b = pred_flat.shape[0]

        for i in range(b):
            valid_indices = torch.where(mask_flat[i])[0]
            if len(valid_indices) < 2:
                continue

            num_to_sample = min(num_samples, len(valid_indices))
            p1_indices = valid_indices[torch.randint(0, len(valid_indices), (num_to_sample,), device=gt_flat.device)]
            p2_indices = valid_indices[torch.randint(0, len(valid_indices), (num_to_sample,), device=gt_flat.device)]

            pred1, pred2 = pred_flat[i, p1_indices], pred_flat[i, p2_indices]
            gt1, gt2 = gt_flat[i, p1_indices], gt_flat[i, p2_indices]

            abs_margin = 0.1  # 至少相差0.1米，过滤掉平坦表面的噪声
            rel_margin_ratio = 0.01  # 至少相差1%，以适应远距离的深度变化

            # 计算相对margin，取两个点深度的平均值作为基准
            avg_depth = (gt1 + gt2) / 2.0
            rel_margin = avg_depth * rel_margin_ratio

            # 两个margin中取较大者，确保近处和远处都有合适的阈值
            dynamic_margin = torch.max(torch.tensor(abs_margin, device=gt1.device), rel_margin)

            # 只有当真值深度差足够大时，才计入loss
            loss_mask = (torch.abs(gt1 - gt2) > dynamic_margin)
            gt_rel = torch.sign(gt1 - gt2)
            #loss_mask = (gt_rel != 0)

            if loss_mask.sum() == 0:
                continue

            pred1_m, pred2_m = pred1[loss_mask], pred2[loss_mask]
            gt_rel_m = gt_rel[loss_mask]
            p1_indices_m, p2_indices_m = p1_indices[loss_mask], p2_indices[loss_mask]


            #loss_pairs = torch.log(1 + torch.exp(-gt_rel_m * (pred1_m - pred2_m)))
            loss_pairs = F.softplus(-gt_rel_m * (pred1_m - pred2_m))

            w1 = weight_flat[i, p1_indices_m]
            w2 = weight_flat[i, p2_indices_m]
            pair_weights = (w1 + w2) / 2.0

            total_weighted_loss += (loss_pairs * pair_weights).sum()
            total_weight += pair_weights.sum()

        if total_weight == 0:
            return torch.tensor(0.0, device=gt_flat.device, requires_grad=True)

        return total_weighted_loss / total_weight

    def forward(self, pred_depth, gt_depth, sampling_mask, weight_map=None, num_samples=512):
        '''
        b, h, w = gt_depth.shape
        pred_flat = pred_depth.reshape(b, -1)
        gt_flat = gt_depth.reshape(b, -1)
        valid_mask = gt_depth > 0.001
        sampling_mask=sampling_mask & valid_mask


        if weight_map is None:
            weight_flat = torch.ones_like(pred_flat)
        else:
            weight_flat = weight_map.reshape(b, -1)

        # ====== 前景掩码采样 ======
        fg_mask = (weight_flat > 0.5) & valid_mask.reshape(b, -1)
        if sampling_mask is not None:
            fg_mask = fg_mask & sampling_mask.reshape(b, -1)

        fg_loss = self.compute_pairwise_loss(pred_flat, gt_flat, weight_flat, fg_mask, self.num_samples)

        # ====== 全局采样 ======

        global_mask = sampling_mask.reshape(b, -1)
        bg_loss = self.compute_pairwise_loss(pred_flat, gt_flat, weight_flat, global_mask, self.num_samples // 2)

        # ====== 融合 ======
        final_loss = self.fg_weight * fg_loss + self.bg_weight * bg_loss
        return final_loss
        '''
        b, h, w = gt_depth.shape
        pred_flat = pred_depth.reshape(b, -1)
        gt_flat = gt_depth.reshape(b, -1)

        if weight_map is None:
            weight_flat = torch.ones_like(pred_flat)
        else:
            weight_flat = weight_map.reshape(b, -1)

        mask_flat = sampling_mask.reshape(b, -1)

        return self.compute_pairwise_loss(
            pred_flat, gt_flat, weight_flat, mask_flat, num_samples
        )
