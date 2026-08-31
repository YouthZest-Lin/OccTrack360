# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

import numpy as np
import os
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import CameraInstance3DBoxes, LiDARInstance3DBoxes

from torch.utils.data import Dataset
from mmdet3d.core.bbox import get_box_type
import copy
import warnings
from mmdet3d.datasets.pipelines import Compose
import mmcv
import torch
import math
from pyquaternion import Quaternion
from tqdm import tqdm
from torch.utils.data import DataLoader

from .metrics import Metric_Occ3d_mIoU, Metric_Occ3d_PQ


def is_abs(path: str) -> bool:
    """Check if path is an absolute path in different backends.

    Args:
        path (str): path of directory or file.

    Returns:
        bool: whether path is an absolute path.
    """
    if osp.isabs(path) or path.startswith(('http://', 'https://', 's3://')):
        return True
    else:
        return False


@DATASETS.register_module()
class OccWaymoDataset(Dataset):
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist'),
        'palette': [
            (0, 120, 255),  # Waymo Blue
            (0, 232, 157),  # Waymo Green
            (255, 205, 85)  # Amber
        ]
    }

    def __init__(
            self,
            data_root: str,
            ann_file: str,
            occupancy_path: str,
            data_prefix: dict = dict(
            pts="training/velodyne",
            CAM_FRONT="training/image_0",
            CAM_FRONT_LEFT="training/image_1",
            CAM_FRONT_RIGHT="training/image_2",
            CAM_SIDE_LEFT="training/image_3",
            CAM_SIDE_RIGHT="training/image_4",
        ),
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_camera=True),
        img_info_prototype="mmcv",
        box_type_3d: str = "LiDAR",
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
        cam_sync_instances: bool = False,
        load_interval: int = 1,
        panoptic=True,
        new_metric=True,
        ignore_index=[9],
        # SOLLOFusion
        use_sequence_group_flag=False,
        sequences_split_num=1,
        file_client_args=dict(backend="disk"),
        inst_class_ids=[1, 2, 3, 4],
        occ_classes=[
                "GO","vehicle","pedestrian","sign","cyclist",
                "trafficlight","pole","constructioncone","bicycle","motorcycle",
                "building","vegetation","treetrunk","road","walkable",
                "free",],
        **kwargs,
    ) -> None:

        super().__init__()
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.occupancy_path = occupancy_path
        self.load_interval = load_interval
        self.img_info_prototype = img_info_prototype
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.inst_class_ids = inst_class_ids
        self.occ_classes = occ_classes
        self.panoptic = panoptic
        self.new_metric = new_metric
        self.ignore_index = ignore_index
        # Join paths.
        self._join_prefix()
        # the argument 'classes' is not used in the code
        self.CLASSES = self.METAINFO["classes"]
        self.cat_ids = range(len(self.METAINFO["classes"]))
        self.cat2id = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.file_client = mmcv.FileClient(**file_client_args)

        # load annotations
        if hasattr(self.file_client, "get_local_path"):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, "rb"))
        else:
            warnings.warn(
                "The used MMCV version does not have get_local_path. "
                f"We treat the {self.ann_file} as local paths and it "
                "might cause errors if the path is not a local path. "
                "Please use MMCV>= 1.3.16 if you meet errors."
            )
            self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.cam_sync_instances = cam_sync_instances
        
        self.pcd_limit_range = pcd_limit_range

        self.filter_empty_gt = filter_empty_gt

        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

        # SOLOFusion
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        if self.test_mode:
            assert self.sequences_split_num == 1
        if self.use_sequence_group_flag: # change the self.flag which means the temporal group
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        else:
            self._set_group_flag()

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        data = mmcv.load(ann_file, file_format="pkl")
        self.metadata = data["metainfo"]
        self.version = self.metadata["version"]
        
        data_infos = []
        for data in data["data_list"]:
            frame_idx = data["sample_idx"] % 1000
            if frame_idx % self.load_interval == 0:
                data_infos.append(data)
        for info in data_infos:
            for cam_key, img_info in info["images"].items():
                cam_prefix = self.data_prefix.get(cam_key, "")
                img_info["img_path"] = osp.join(cam_prefix, img_info["img_path"])

            pts_prefix = self.data_prefix["pts"]
            info["lidar_points"]["lidar_path"] = osp.join(
                pts_prefix, info["lidar_points"]["lidar_path"]
            )
        
        return data_infos

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None

        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]  # from .pkl
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            index=index,
            sample_idx=info["sample_idx"],
            pts_filename=info["lidar_points"]["lidar_path"],
            ego2global=info["ego2global"],
            timestamp=info["timestamp"]/ 1e6,  # now, the unit is second.
            anno_infos=info['cam_sync_instances'],
        )

        if self.modality["use_camera"]:
            assert "bevdet" in self.img_info_prototype
            input_dict.update(dict(curr=info))

            if self.use_sequence_group_flag:
                input_dict["sample_index"] = index
                input_dict["sequence_group_idx"] = self.flag[index]
                input_dict["start_of_sequence"] = (
                    index == 0 or self.flag[index - 1] != self.flag[index]
                )
                # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
                # if they belong to same sequence.
                ego2global = np.array(self.data_infos[index]["ego2global"])
                input_dict["waymo_get_rt_matrix"] = dict(
                    ego2global_rotation=ego2global[:3, :3],
                    ego2global_translation=ego2global[:3, 3],
                )
                if not input_dict["start_of_sequence"]:
                    input_dict["curr_to_prev_ego_rt"] = torch.FloatTensor(
                        self.waymo_get_rt_matrix(
                            self.data_infos[index],
                            self.data_infos[index - 1],
                            "ego",
                            "ego",
                        )
                    )
                    input_dict["prev_ego_to_global_rt"] = torch.FloatTensor(
                        self.waymo_get_rt_matrix(
                            self.data_infos[index - 1],
                            self.data_infos[index],
                            "ego",
                            "global",
                        )
                    )  # global is same for all in one sequence.

                else:
                    input_dict["curr_to_prev_ego_rt"] = torch.eye(4).float()
                    input_dict["prev_ego_to_global_rt"] = torch.FloatTensor(
                        self.waymo_get_rt_matrix(
                            self.data_infos[index],
                            self.data_infos[index],
                            "ego",
                            "global",
                        )
                    )

                input_dict["global_to_curr_ego_rt"] = torch.FloatTensor(
                    self.waymo_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index], "global", "ego"
                    )
                )

        return input_dict

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        """
        Set flag
        """
        self.flag = np.arange(len(self), dtype=np.int64)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """

        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            frame_idx = self.data_infos[idx]["sample_idx"] % 1000
            if idx != 0 and frame_idx - self.load_interval < 0:
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag] / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )
                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
        """
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError("prefix should be a string, but got " f"{type(prefix)}")
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = os.path.join(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.eval_ray = False
        if self.eval_ray:
            return self.evaluate_ray_metric(
                occ_results, runner, show_dir, **eval_kwargs
            )
        else:
            return self.evaluate_voxel_metric(
                occ_results, runner, show_dir, **eval_kwargs
            )

    def evaluate_voxel_metric(
        self, occ_results, runner=None, show_dir=None, save=False, **eval_kwargs
    ):
        if show_dir is not None:
            mmcv.mkdir_or_exist(show_dir)
            mmcv.mkdir_or_exist(os.path.join(show_dir, "occupancy_pred"))
            print("\nSaving output and gt in {} for visualization.".format(show_dir))

        self.occ_sem_metrics = Metric_Occ3d_mIoU(
            num_classes=len(self.occ_classes), 
            use_lidar_mask=False, 
            use_image_mask=True,
            ignore_index=self.ignore_index,
        )
        if self.panoptic:
            self.occ_pano_metrics = Metric_Occ3d_PQ(
                num_classes=len(self.occ_classes),
                inst_class_ids=self.inst_class_ids,
                use_lidar_mask=False,
                use_image_mask=True,
                ignore_index=self.ignore_index,
            )
        
        print("\nStarting Evaluation...")
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):
            if self.panoptic:
                semantics_pred = occ_pred_w_index["pano_sem"].reshape((200, 200, 16))
                # semantics_pred = occ_pred_w_index["sem_pred"].reshape((200, 200, 16))
                instances_pred = occ_pred_w_index["pano_inst"].reshape((200, 200, 16))
            else:
                semantics_pred = occ_pred_w_index["sem_pred"].reshape((200, 200, 16))
            index = occ_pred_w_index["index"]
            if index in processed_set:
                continue
            processed_set.add(index)

            info = self.data_infos[index]
            pts_filename = info["lidar_points"]["lidar_path"].split("/")[-1]
            basename = os.path.basename(pts_filename)
            seq_name = basename[1:4]
            frame_name = basename[4:7]

            split_set = "validation" if "train" not in self.ann_file else "training"
            occ_path = os.path.join(
                self.occupancy_path, split_set, seq_name, "{}_04.npz".format(frame_name)
            )
            occ_gt = np.load(occ_path)

            semantics_gt = occ_gt["semantics"]
            instances_gt = occ_gt["instances"]
            mask_lidar = None
            mask_camera = occ_gt["mask_camera"].astype(bool) & occ_gt["infov"].astype(bool)

            self.occ_sem_metrics.add_batch(
                semantics_pred, semantics_gt, mask_lidar, mask_camera
            )
            if self.panoptic:
                self.occ_pano_metrics.add_batch(
                    semantics_pred,
                    semantics_gt,
                    instances_pred,
                    instances_gt,
                    mask_lidar,
                    mask_camera,
                )
        res = {}
        res["semantic"] = self.occ_sem_metrics.count_miou()
        if self.panoptic:
            res["panoptic"] = self.occ_pano_metrics.count_pq()
        return res  # we need a dict(), eval_res

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info["token"]
            save_path = os.path.join(submission_prefix, "{}.npz".format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print("\nFinished.")


    def waymo_get_rt_matrix(self, src_sample, dest_sample, src_mod, dest_mod):
        """
        CAM_FRONT_XYD indicates going from 2d image coords + depth
            Note that image coords need to multiplied with said depths first to bring it into 2d hom coords.
        CAM_FRONT indicates going from camera coordinates xyz

        Method is: whatever the input is, transform to global first.
        """
        possible_mods = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "lidar",
            "ego",
            "global",
        ]

        assert src_mod in possible_mods and dest_mod in possible_mods

        src_ego_to_global = np.eye(4, 4)
        src_ego_to_global = np.array(src_sample["ego2global"])

        dest_ego_to_global = np.array(dest_sample["ego2global"])

        src_mod_to_global = None
        dest_global_to_mod = None

        if src_mod == "global":
            src_mod_to_global = np.eye(4, 4)
        elif src_mod == "ego":
            src_mod_to_global = src_ego_to_global
        
        if dest_mod == "global":
            dest_global_to_mod = np.eye(4, 4)
        elif dest_mod == "ego":
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global)

        return dest_global_to_mod @ src_mod_to_global
