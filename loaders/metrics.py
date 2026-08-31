import os
import numpy as np
from sklearn.neighbors import KDTree
from termcolor import colored
from functools import reduce
from typing import Iterable
from scipy.optimize import linear_sum_assignment

class Metric_Occ3d_mIoU:
    def __init__(
        self,
        save_dir=".",
        num_classes=18,
        ignore_index=[],
        use_lidar_mask=False,
        use_image_mask=False,
        point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        occupancy_size=[0.4, 0.4, 0.4],
    ):
        if num_classes == 18:  # nuScenes
            self.class_names = [
                'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation','free'
            ]
        elif num_classes == 16:  # Waymo
            self.class_names = [
                'GO', 'vehicle', 'pedestrian', 'sign', 'cyclist',
                'trafficlight', 'pole', 'constructioncone', 'bicycle', 'motorcycle',
                'building', 'vegetation', 'treetrunk','road', 'walkable', 'free'
            ]
        elif num_classes == 2:  # binary
            self.class_names = ["non-free", "free"]

        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.point_cloud_range = point_cloud_range
        self.occupancy_size = occupancy_size
        self.occ_xdim = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0])
            / self.occupancy_size[0]
        )
        self.occ_ydim = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1])
            / self.occupancy_size[1]
        )
        self.occ_zdim = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2])
            / self.occupancy_size[2]
        )
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl**2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        # return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        result = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        result[hist.sum(1) == 0] = float("nan")
        return result

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(
            n_classes, pred.flatten(), label.flatten()
        )
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        if self.num_classes == 2:
            masked_semantics_pred = np.copy(masked_semantics_pred)
            masked_semantics_gt = np.copy(masked_semantics_gt)
            masked_semantics_pred[masked_semantics_pred < 17] = 0
            masked_semantics_pred[masked_semantics_pred == 17] = 1
            masked_semantics_gt[masked_semantics_gt < 17] = 0
            masked_semantics_gt[masked_semantics_gt == 17] = 1

        _, _hist = self.compute_mIoU(
            masked_semantics_pred, masked_semantics_gt, self.num_classes
        )
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        res = {} # log
        new_mIoU = []
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            if ind_class in self.ignore_index:
                continue
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            new_mIoU.append(mIoU[ind_class])
            res[self.class_names[ind_class]] = round(mIoU[ind_class] * 100, 2)
        
        new_mIoU = np.array(new_mIoU)
        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(new_mIoU) * 100, 2)))
        res['Overall_mIoU'] = round(np.nanmean(new_mIoU) * 100, 2)

        return res





# modified from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/evaluation/functional/panoptic_seg_eval.py#L10
class Metric_Occ3d_PQ:
    def __init__(
        self,
        save_dir=".",
        num_classes=18,
        inst_class_ids=None,
        use_lidar_mask=False,
        use_image_mask=False,
        ignore_index: Iterable[int] = [],
        min_num_points=10,
    ):
        """
        Args:
            ignore_index (llist): Class ids that not be considered in pq counting.
        """
        if num_classes == 18:
            self.class_names = [
                'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation','free'
            ]
            # ['bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'trailer', 'truck']
            if inst_class_ids is not None:
                self.inst_class_ids = inst_class_ids
            else:
                self.inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]
        elif num_classes == 16:
            self.class_names = [
                'GO', 'vehicle', 'pedestrian', 'sign', 'cyclist',
                'trafficlight', 'pole', 'constructioncone', 'bicycle', 'motorcycle',
                'building', 'vegetation', 'treetrunk','road', 'walkable', 'free'
            ]
            if inst_class_ids is not None:
                self.inst_class_ids = inst_class_ids
            else:
                self.inst_class_ids = [1, 2, 3, 4]
        else:
            raise ValueError

        self.save_dir = save_dir
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.ignore_index = ignore_index
        self.id_offset = 2**16
        self.eps = 1e-5

        self.min_num_points = (
            min_num_points
        )
        self.include = np.array(
            [n for n in range(self.num_classes - 1) if n not in self.ignore_index],
            dtype=int,
        )
        self.cnt = 0

        # panoptic stuff
        self.pan_tp = np.zeros(self.num_classes, dtype=int)
        self.pan_iou = np.zeros(self.num_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.num_classes, dtype=int)
        self.pan_fn = np.zeros(self.num_classes, dtype=int)

    def add_batch(
        self,
        semantics_pred,
        semantics_gt,
        instances_pred,
        instances_gt,
        mask_lidar,
        mask_camera,
    ):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
            masked_instances_gt = instances_gt[mask_camera]
            masked_instances_pred = instances_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
            masked_instances_gt = instances_gt[mask_lidar]
            masked_instances_pred = instances_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
            masked_instances_gt = instances_gt
            masked_instances_pred = instances_pred
        self.add_panoptic_sample(
            masked_semantics_pred,
            masked_semantics_gt,
            masked_instances_pred,
            masked_instances_gt,
        )

    def add_panoptic_sample(
        self, semantics_pred, semantics_gt, instances_pred, instances_gt
    ):
        """Add one sample of panoptic predictions and ground truths for
        evaluation.

        Args:
            semantics_pred (np.ndarray): Semantic predictions.
            semantics_gt (np.ndarray): Semantic ground truths.
            instances_pred (np.ndarray): Instance predictions.
            instances_gt (np.ndarray): Instance ground truths.
        """
        # get instance_class_id from instance_gt
        instance_class_ids = [self.num_classes - 1]
        instance_ids = []
        unique_inst_ids = np.unique(instances_gt.reshape(-1))
        for i in unique_inst_ids:
            if i == 0:
                continue
            class_id = np.unique(semantics_gt[instances_gt == i])
            # assert class_id.shape[0] == 1, "each instance must belong to only one class"
            if class_id.shape[0] == 1:
                instance_class_ids.append(class_id[0])
                instance_ids.append(i)
            else:
                instance_class_ids.append(self.num_classes - 1)
                instance_ids.append(i)

        instance_count = 1
        final_instance_class_ids = []
        final_instances = np.zeros_like(instances_gt)  # empty space has instance id "0"

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics_gt == class_id) == 0:
                continue

            if class_id in self.inst_class_ids:
                # treat as instances
                for index, instance_id in enumerate(instance_ids):
                    if instance_class_ids[index] != class_id:
                        continue
                    final_instances[instances_gt == instance_id] = instance_count
                    instance_count += 1
                    final_instance_class_ids.append(class_id)
            else:
                # treat as semantics
                final_instances[semantics_gt == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)

        instances_gt = final_instances

        for cl in self.ignore_index:
            # make a mask for this class
            gt_not_in_excl_mask = semantics_gt != cl
            # remove all other points
            semantics_pred = semantics_pred[gt_not_in_excl_mask]
            semantics_gt = semantics_gt[gt_not_in_excl_mask]
            instances_pred = instances_pred[gt_not_in_excl_mask]
            instances_gt = instances_gt[gt_not_in_excl_mask]

        # for each class (except the ignored ones)
        for cl in self.include:
            # get a class mask
            pred_inst_in_cl_mask = semantics_pred == cl
            gt_inst_in_cl_mask = semantics_gt == cl

            # get instance points in class (makes outside stuff 0)
            pred_inst_in_cl = instances_pred * pred_inst_in_cl_mask.astype(int)
            gt_inst_in_cl = instances_gt * gt_inst_in_cl_mask.astype(int)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                pred_inst_in_cl[pred_inst_in_cl > 0], return_counts=True
            )
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                gt_inst_in_cl[gt_inst_in_cl > 0], return_counts=True
            )
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])

            # define a iou matrix btw pred and gt
            iou_matrix = np.zeros((unique_pred.shape[0], unique_gt.shape[0]))

            # generate intersection using offset
            valid_combos = np.logical_and(pred_inst_in_cl > 0, gt_inst_in_cl > 0)
            id_offset_combo = (
                pred_inst_in_cl[valid_combos]
                + self.id_offset * gt_inst_in_cl[valid_combos]
            )
            unique_combo, counts_combo = np.unique(id_offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.id_offset
            pred_labels = unique_combo % self.id_offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(float) / unions.astype(float)
            # use the ious to fill the matrix
            for i, j, iou in zip(pred_labels, gt_labels, ious):
                iou_matrix[id2idx_pred[i], id2idx_gt[j]] = iou
            # use the Hungarian algorithm to get the best matches
            pred_indexes, gt_indexes = linear_sum_assignment(-iou_matrix)
            # get tp_indexes by iou > 0
            matched_ious = iou_matrix[pred_indexes, gt_indexes]
            tp_indexes = matched_ious > 0

            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(matched_ious[tp_indexes])

            matched_gt[gt_indexes[tp_indexes]] = True
            matched_pred[pred_indexes[tp_indexes]] = True
            # matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            # matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True
            # count the FN
            if len(counts_gt) > 0:
                self.pan_fn[cl] += np.sum(
                    np.logical_and(counts_gt >= self.min_num_points,
                                   ~matched_gt))

            # count the FP
            if len(matched_pred) > 0:
                self.pan_fp[cl] += np.sum(
                    np.logical_and(counts_pred >= self.min_num_points,
                                   ~matched_pred))
    
    def count_pq(self, ):
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double), self.eps)
        pq_all = sq_all * rq_all

        # mask classes not occurring in dataset
        mask = (self.pan_tp + self.pan_fp + self.pan_fn) > 0
        sq_all[~mask] = float("nan")
        rq_all[~mask] = float("nan")
        pq_all[~mask] = float("nan")

        # then do the REAL mean (no ignored classes)
        sq = round(np.nanmean(sq_all[self.include]) * 100, 2)
        rq = round(np.nanmean(rq_all[self.include]) * 100, 2)
        pq = round(np.nanmean(pq_all[self.include]) * 100, 2)

        print(f"===> per class sq, rq, pq of {self.cnt} samples:")
        res = {}
        for ind_class in self.include:
            print(
                f"===> {self.class_names[ind_class]} -"
                + f" sq = {round(sq_all[ind_class] * 100, 2)},"
                + f" rq = {round(rq_all[ind_class] * 100, 2)},"
                + f" pq = {round(pq_all[ind_class] * 100, 2)}"
            )

            res[self.class_names[ind_class]] = (
                round(pq_all[ind_class] * 100, 2),
                round(sq_all[ind_class] * 100, 2),
                round(rq_all[ind_class] * 100, 2),
            )

        print(f"===> sq of {self.cnt} samples: " + str(sq))
        print(f"===> rq of {self.cnt} samples: " + str(rq))
        print(f"===> pq of {self.cnt} samples: " + str(pq))
        res["Overall_pq_sq_rq"] = (pq, sq, rq)
        return res