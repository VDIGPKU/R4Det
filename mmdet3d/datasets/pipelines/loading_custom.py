import json

import numpy as np
from matplotlib import pyplot as plt
from mmdet.datasets.builder import PIPELINES
import os
import cv2
from mmcv.utils import build_from_cfg
import torch
from skimage.draw import polygon as draw_polygon

from mmdet3d.datasets.structures.mask import BitmapMasks


@PIPELINES.register_module()
class LoadMyDepthFromFile(object):
    """Load custom H*W depth map from a .npy file."""
    def __init__(self, depth_base_path):
        self.depth_base_path = depth_base_path

    def __call__(self, results):
        # Construct the full path to the .npy file
        # 'sample_idx' is usually the filename without extension (e.g., '000123')
        sample_idx = results['sample_idx']
        formatted_idx = f"{int(sample_idx):06d}"
        depth_map_path = os.path.join(self.depth_base_path, f"{formatted_idx}.npy")

        my_depth_map = np.load(depth_map_path).astype(np.float32)

        # Add the loaded map to the results dictionary under a new key
        results['my_gt_depth'] = my_depth_map
        my_depth_map = torch.from_numpy(my_depth_map).float()

        #exit(0)
        return results
@PIPELINES.register_module()
class LoadMyDepthFromFilevod(object):
    """Load custom H*W depth map from a .npy file."""
    def __init__(self, depth_base_path):
        self.depth_base_path = depth_base_path

    def __call__(self, results):
        # Construct the full path to the .npy file
        # 'sample_idx' is usually the filename without extension (e.g., '000123')
        sample_idx = results['sample_idx']
        formatted_idx = f"{int(sample_idx):05d}"
        depth_map_path = os.path.join(self.depth_base_path, f"{formatted_idx}.npy")

        try:
            # Load the numpy array and ensure it's float32
            my_depth_map = np.load(depth_map_path).astype(np.float32)
        except FileNotFoundError:
            print(f"\nWarning: Custom depth map not found: {depth_map_path}, using a zero map.")
            img_shape = results['img_shape'] # H, W, C
            my_depth_map = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)

        # Add the loaded map to the results dictionary under a new key
        results['my_gt_depth'] = my_depth_map
        my_depth_map = torch.from_numpy(my_depth_map).float()

        #exit(0)
        return results
@PIPELINES.register_module()
class DownsampleDepthMap(object):
    """Downsamples the depth map to the target feature map size."""
    def __init__(self, target_shape=(60, 80)):
        self.target_shape = target_shape
        self.dsize = (target_shape[1], target_shape[0])

    def __call__(self, results):
        if 'my_gt_depth' in results:
            depth_map = results['my_gt_depth']
            downsampled_depth = cv2.resize(
                depth_map,
                dsize=self.dsize,
                interpolation=cv2.INTER_NEAREST
            )
            results['my_gt_depth'] = downsampled_depth
        return results


@PIPELINES.register_module()
class BoxesFromSegMask(object):


    def __init__(self, min_area_threshold=100):
        self.min_area_threshold = min_area_threshold
    def __call__(self, results):
        if 'segmentation' not in results:
            results['derived_2d_boxes'] = np.empty((0, 4), dtype=np.float32)
            return results

        seg_mask = results['segmentation']
        if seg_mask.ndim == 3 and seg_mask.shape[2] == 3:
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
        binary_mask = (seg_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        derived_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_threshold:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            derived_boxes.append([x, y, x + w, y + h])

        if not derived_boxes:
            results['derived_2d_boxes'] = np.empty((0, 4), dtype=np.float32)
        else:
            results['derived_2d_boxes'] = np.array(derived_boxes, dtype=np.float32)
        return results


import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES
import os
import cv2
from skimage.draw import rectangle, rectangle_perimeter


@PIPELINES.register_module()
class GenerateBevFromBoxes(object):

    def __init__(self, box_key, point_cloud_range, grid_size,
                 class_sizes,
                 depth_heuristic,
                 save_vis=False):
        self.box_key = box_key
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.grid_size = np.array(grid_size, dtype=np.int32)

        # e.g., {'Car': [4.5, 1.8], 'Pedestrian': [0.8, 0.8]} # L, W in meters
        self.class_sizes = class_sizes

        # e.g., {'y_max': 900, 'd_min': 2.0, 'd_max': 50.0}
        self.depth_heuristic = depth_heuristic

        self.scale_x = self.grid_size[0] / (self.point_cloud_range[3] - self.point_cloud_range[0])
        self.scale_y = self.grid_size[1] / (self.point_cloud_range[4] - self.point_cloud_range[1])

        self.save_vis = save_vis
        if self.save_vis:
            self.debug_vis_path = '/data/tangyousen/bevperception/bev_box_final'

    def estimate_depth(self, v_bottom):
        y_max = self.depth_heuristic['y_max']
        d_min = self.depth_heuristic['d_min']
        d_max = self.depth_heuristic['d_max']
        depth = d_max - (v_bottom / y_max) * (d_max - d_min)
        return max(d_min, min(d_max, depth))

    def __call__(self, results):
        bev_shape = (self.grid_size[1], self.grid_size[0])
        bev_semantic_mask = np.zeros(bev_shape, dtype=np.float32)

        if self.box_key not in results or len(results[self.box_key]) == 0:
            results['bev_semantic_mask'] = torch.from_numpy(bev_semantic_mask)
            return results

        cam2img = results['cam2img']
        lidar2cam = results['lidar2cam']
        cam_intrinsics_inv = np.linalg.inv(cam2img[:3, :3])
        lidar2cam_inv = np.linalg.inv(lidar2cam)

        boxes_2d = results[self.box_key]
        labels = results['gt_labels']
        class_names = results['gt_names']

        if self.save_vis:
            bev_canvas = np.zeros((bev_shape[0], bev_shape[1], 3), dtype=np.uint8)
            cv2.rectangle(bev_canvas, (0, 0), (bev_shape[1] - 1, bev_shape[0] - 1), (0, 255, 0), 1)

        for box2d, label, name in zip(boxes_2d, labels, class_names):
            if name not in self.class_sizes:
                continue

            x1, y1, x2, y2 = box2d

            u_center = (x1 + x2) / 2
            v_bottom = y2
            est_depth = self.estimate_depth(v_bottom)

            point_img_hom = np.array([u_center * est_depth, v_bottom * est_depth, est_depth])
            point_cam = cam_intrinsics_inv @ point_img_hom
            point_cam_hom = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
            point_lidar = (lidar2cam_inv @ point_cam_hom)[:3]

            bev_center_world_x, bev_center_world_y = point_lidar[0], point_lidar[1]
            L, W = self.class_sizes[name]

            bev_box_world = np.array([
                [bev_center_world_x - L / 2, bev_center_world_y - W / 2],
                [bev_center_world_x + L / 2, bev_center_world_y - W / 2],
                [bev_center_world_x + L / 2, bev_center_world_y + W / 2],
                [bev_center_world_x - L / 2, bev_center_world_y + W / 2],
            ])
            bev_box_grid = np.zeros_like(bev_box_world)
            bev_box_grid[:, 0] = (bev_box_world[:, 0] - self.point_cloud_range[0]) * self.scale_x
            bev_box_grid[:, 1] = (bev_box_world[:, 1] - self.point_cloud_range[1]) * self.scale_y

            bev_box_grid[:, 0] = np.clip(bev_box_grid[:, 0], 0, bev_shape[1] - 1)
            bev_box_grid[:, 1] = np.clip(bev_box_grid[:, 1], 0, bev_shape[0] - 1)

            start_point = (int(bev_box_grid[0, 0]), int(bev_box_grid[0, 1]))
            end_point = (int(bev_box_grid[2, 0]), int(bev_box_grid[2, 1]))
            rr, cc = rectangle(start_point, end_point, shape=bev_shape)
            bev_semantic_mask[rr, cc] = 1.0

            if self.save_vis:
                cv2.rectangle(bev_canvas, start_point, end_point, (0, 0, 255), 1)

        results['bev_semantic_mask'] = torch.from_numpy(bev_semantic_mask)

        bev_canvas[bev_semantic_mask > 0] = [255, 255, 255]
        os.makedirs(self.debug_vis_path, exist_ok=True)
        sample_idx = results.get('sample_idx', 'unknown')
        save_path = os.path.join(self.debug_vis_path, f'bev_box_mask_{sample_idx}.png')

        bev_canvas_rgb = cv2.cvtColor(bev_canvas, cv2.COLOR_BGR2RGB)
        bev_canvas_rgb = cv2.flip(bev_canvas_rgb, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(bev_canvas_rgb)
        plt.title(f"BEV Box Mask - {sample_idx}\nGreen: Range, Red: Box Outlines, White: Final Mask")
        plt.xlabel("BEV Grid X")
        plt.ylabel("BEV Grid Y")
        plt.grid(False)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        return results

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
@PIPELINES.register_module()
class LoadVLSAMAnnotations(object):

    def __init__(self,
                 ann_root_path='/data/tangyousen/TJ4DRadSet_4DRadar/annotations/',
                 mask_root_path='/data/tangyousen/TJ4DRadSet_4DRadar/',
                 class_map={'Pedestrian': 0, 'Cyclist': 1, 'Car': 2, 'Truck': 3},
                 iou_threshold=0.1):
        self.ann_root_path = ann_root_path
        self.mask_root_path = mask_root_path
        self.class_map = class_map
        self.iou_threshold = iou_threshold
        print(f"Initialized Custom LoadVLSAMAnnotations with IoU threshold: {self.iou_threshold}")

    def __call__(self, results):
        debug_save_path='/data/tangyousen/TJ4DRadSet_4DRadar/debug_matched_masks/'
        if 'gt_bboxes' not in results or 'gt_labels' not in results:
            raise KeyError("Official 'gt_bboxes' and 'gt_labels' not found in results. "
                           "Ensure LoadAnnotations3D runs before this pipeline step and with_bbox/with_label is True.")

        official_bboxes = results['gt_bboxes']
        official_labels = results['gt_labels']
        num_official_gts = len(official_bboxes)
        h, w = results['img_shape'][:2]
        if num_official_gts == 0:
            results['gt_masks'] = BitmapMasks(np.zeros((0, h, w), dtype=np.uint8), h, w)
            if 'mask_fields' not in results: results['mask_fields'] = []
            results['mask_fields'].append('gt_masks')
            return results

        img_filename = results['filename']
        base_filename = os.path.splitext(os.path.basename(img_filename))[0]
        json_path = os.path.join(self.ann_root_path, f'{base_filename}.json')

        vlsam_masks_list, vlsam_labels_list, vlsam_bboxes_list = [], [], []

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = []

            for ann in annotations:
                class_name = ann.get('class_name')
                mask_relative_path = ann.get('mask_path')
                if not class_name or not mask_relative_path or class_name not in self.class_map: continue
                label = self.class_map[class_name]
                full_mask_path = os.path.join(self.mask_root_path, mask_relative_path)
                mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None: continue
                mask = (mask_img > 128).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w_b, h_b = cv2.boundingRect(main_contour)
                vlsam_labels_list.append(label)
                vlsam_bboxes_list.append([x, y, x + w_b, y + h_b])
                vlsam_masks_list.append(mask)

        vlsam_labels = np.array(vlsam_labels_list, dtype=np.int64)
        vlsam_bboxes = np.array(vlsam_bboxes_list, dtype=np.float32).reshape(-1, 4)
        num_vlsam_preds = len(vlsam_bboxes)
        aligned_masks_list = []
        if num_vlsam_preds > 0:
            iou_matrix = bbox_overlaps(official_bboxes, vlsam_bboxes)

            vlsam_mask_used = [False] * num_vlsam_preds

            for i in range(num_official_gts):
                gt_label = official_labels[i]
                best_iou = -1.0
                best_match_idx = -1

                for j in range(num_vlsam_preds):
                    if gt_label == vlsam_labels[j]:
                        if iou_matrix[i, j] > best_iou:
                            best_iou = iou_matrix[i, j]
                            best_match_idx = j

                is_match_found = best_match_idx != -1
                is_iou_high = is_match_found #and best_iou >= self.iou_threshold
                is_available = is_match_found and not vlsam_mask_used[best_match_idx]



                #if is_match_found and is_iou_high and is_available:

                matched_mask = vlsam_masks_list[best_match_idx]
                aligned_masks_list.append(matched_mask)
                vlsam_mask_used[best_match_idx] = True
                if debug_save_path:
                    sample_save_dir = os.path.join(debug_save_path, base_filename)
                    os.makedirs(sample_save_dir, exist_ok=True)
                    save_filename = f"gt_{i}_matched_to_vlsam_{best_match_idx}.png"
                    cv2.imwrite(os.path.join(sample_save_dir, save_filename), matched_mask * 255)
                #else:
                #    aligned_masks_list.append(np.zeros((h, w), dtype=np.uint8))
        else:
            for _ in range(num_official_gts):
                aligned_masks_list.append(np.zeros((h, w), dtype=np.uint8))
        final_gt_masks_np = np.stack(aligned_masks_list, axis=0) if aligned_masks_list else np.zeros((0, h, w),
                                                                                                     dtype=np.uint8)
        results['gt_masks'] = BitmapMasks(final_gt_masks_np, h, w)
        if 'mask_fields' not in results: results['mask_fields'] = []
        if 'gt_masks' not in results['mask_fields']: results['mask_fields'].append('gt_masks')


        return results

@PIPELINES.register_module()
class LoadVLSAMAnnotationsvod(object):

    def __init__(self,
                 ann_root_path='/data/tangyousen/view-of-delft_PUBLIC/view_of_delft_PUBLIC/lidar/training/annotations/',
                 mask_root_path='/data/tangyousen/view-of-delft_PUBLIC/view_of_delft_PUBLIC/lidar/training/',
                 class_map={'Pedestrian': 0, 'Cyclist': 1, 'Car': 2},
                 iou_threshold=0.1):
        self.ann_root_path = ann_root_path
        self.mask_root_path = mask_root_path
        self.class_map = class_map
        self.iou_threshold = iou_threshold
        print(f"Initialized Custom LoadVLSAMAnnotations with IoU threshold: {self.iou_threshold}")

    def __call__(self, results):
        debug_save_path='/data/tangyousen/view-of-delft_PUBLIC/view_of_delft_PUBLIC/radar_5frames/debug_matched_masks/'

        if 'gt_bboxes' not in results or 'gt_labels' not in results:
            raise KeyError("Official 'gt_bboxes' and 'gt_labels' not found in results. "
                           "Ensure LoadAnnotations3D runs before this pipeline step and with_bbox/with_label is True.")

        official_bboxes = results['gt_bboxes']
        official_labels = results['gt_labels']
        num_official_gts = len(official_bboxes)
        h, w = results['img_shape'][:2]
        if num_official_gts == 0:
            results['gt_masks'] = BitmapMasks(np.zeros((0, h, w), dtype=np.uint8), h, w)
            if 'mask_fields' not in results: results['mask_fields'] = []
            results['mask_fields'].append('gt_masks')
            return results

        img_filename = results['filename']
        base_filename = os.path.splitext(os.path.basename(img_filename))[0]
        json_path = os.path.join(self.ann_root_path, f'{base_filename}.json')

        vlsam_masks_list, vlsam_labels_list, vlsam_bboxes_list = [], [], []

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = []

            for ann in annotations:
                class_name = ann.get('class_name')
                mask_relative_path = ann.get('mask_path')
                if not class_name or not mask_relative_path or class_name not in self.class_map: continue
                label = self.class_map[class_name]
                full_mask_path = os.path.join(self.mask_root_path, mask_relative_path)
                mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None: continue
                mask = (mask_img > 128).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w_b, h_b = cv2.boundingRect(main_contour)
                vlsam_labels_list.append(label)
                vlsam_bboxes_list.append([x, y, x + w_b, y + h_b])
                vlsam_masks_list.append(mask)

        vlsam_labels = np.array(vlsam_labels_list, dtype=np.int64)
        vlsam_bboxes = np.array(vlsam_bboxes_list, dtype=np.float32).reshape(-1, 4)
        num_vlsam_preds = len(vlsam_bboxes)

        aligned_masks_list = []
        if num_vlsam_preds > 0:
            iou_matrix = bbox_overlaps(official_bboxes, vlsam_bboxes)

            vlsam_mask_used = [False] * num_vlsam_preds

            for i in range(num_official_gts):
                gt_label = official_labels[i]
                best_iou = -1.0
                best_match_idx = -1

                for j in range(num_vlsam_preds):
                    if gt_label == vlsam_labels[j]:
                        if iou_matrix[i, j] > best_iou:
                            best_iou = iou_matrix[i, j]
                            best_match_idx = j

                is_match_found = best_match_idx != -1
                is_iou_high = is_match_found #and best_iou >= self.iou_threshold
                is_available = is_match_found and not vlsam_mask_used[best_match_idx]



                #if is_match_found and is_iou_high and is_available:

                matched_mask = vlsam_masks_list[best_match_idx]
                aligned_masks_list.append(matched_mask)
                vlsam_mask_used[best_match_idx] = True
                if debug_save_path:
                    sample_save_dir = os.path.join(debug_save_path, base_filename)
                    os.makedirs(sample_save_dir, exist_ok=True)
                    save_filename = f"gt_{i}_matched_to_vlsam_{best_match_idx}.png"
                    cv2.imwrite(os.path.join(sample_save_dir, save_filename), matched_mask * 255)
                #else:
                #    aligned_masks_list.append(np.zeros((h, w), dtype=np.uint8))
        else:
            for _ in range(num_official_gts):
                aligned_masks_list.append(np.zeros((h, w), dtype=np.uint8))

        final_gt_masks_np = np.stack(aligned_masks_list, axis=0) if aligned_masks_list else np.zeros((0, h, w),
                                                                                                     dtype=np.uint8)
        results['gt_masks'] = BitmapMasks(final_gt_masks_np, h, w)
        if 'mask_fields' not in results: results['mask_fields'] = []
        if 'gt_masks' not in results['mask_fields']: results['mask_fields'].append('gt_masks')


        #exit(0)
        return results