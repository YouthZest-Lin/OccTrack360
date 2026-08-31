# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np
import os
from mmdet.datasets import DATASETS
from .waymo_occ_dataset import OccWaymoDataset
from torch.utils.data import Dataset
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
import mmcv
import torch
import math
from tqdm import tqdm
from .track_metrics import OccPanoptic4DEval, OccPanoptic4DEvalDetail

@DATASETS.register_module()
class WindowTrackOccWaymoDataset(OccWaymoDataset):
    def __init__(self, 
                 pipeline_post=None, 
                 sample_mode='fixed_interval',
                 sample_interval=1,
                 num_frames_per_sample=3,
                 use_detail_eval=False,
                 **kwargs):
        super(WindowTrackOccWaymoDataset, self).__init__(**kwargs)
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.num_frames_per_sample = num_frames_per_sample

        if pipeline_post is not None:
            self.pipeline_post = Compose(pipeline_post)

        self.use_detail_eval = use_detail_eval

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

    def _rand_another(self, idx):
        """Randomly get another item with different flag.

        Returns:
            int: Another index of item with different flag.
        """
        pool = np.where(self.flag != self.flag[idx])[0]
        return np.random.choice(pool)

    def _get_sample_range(self, start_idx):
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        
        index_frame_num = self.data_infos[start_idx]["sample_idx"] # 0 000 000 : train/test sequence_num frame_num
        index_sequnce_num = index_frame_num // 1000 % 1000
        # determine the end index
        end_idx = start_idx + sample_interval
        i = 1
        while end_idx < len(self.data_infos) and  i < self.num_frames_per_sample:   
            if self.data_infos[end_idx]["sample_idx"] // 1000 % 1000 != index_sequnce_num:
                break
            i += 1
            end_idx += sample_interval

        default_range = start_idx, end_idx, sample_interval
        return default_range
    
    def prepare_train_data(self, index):
        start, end, interval = self._get_sample_range(index)

        if len(range(start, end, interval)) < self.num_frames_per_sample:
            return None 
        
        ret = None
        for i in range(start, end, interval):
            data_i = self.prepare_train_data_single(i)
            if data_i is None:
                return None

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)
        
        ret = self.pipeline_post(ret)
        return ret

    def prepare_train_data_single(self, index):
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
            example is None or (example["instid_and_clsid"]._data.size(0) == 0)
        ):
            return None

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
            timestamp=info["timestamp"]/ 1e6,  # after 1e6, the unit is second.
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
                    )

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

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        assert self.use_sequence_group_flag == True
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def flatten(self, nested_list):
        for element in nested_list:
            if isinstance(element, list):
                yield from self.flatten(element)
            else:
                yield element

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
        self, occ_results, runner=None, show_dir=None, save=False, model_name='temp', save_path='./statistics', **eval_kwargs
    ):# occ_results is in order of the dataset
        if show_dir is not None:
            mmcv.mkdir_or_exist(show_dir)
            mmcv.mkdir_or_exist(os.path.join(show_dir, "occupancy_pred"))
            print("\nSaving output and gt in {} for visualization.".format(show_dir))
        # occ_result is a list:
        # {'index': 0, 'sample_idx': 0, 'pano_inst': array([0, ...], dtype=int16), 'pano_sem': array([15, ...], dtype=uint8)}
        free_label = len(self.occ_classes) - 1
        ignore = self.ignore_index + [free_label] if free_label not in self.ignore_index else self.ignore_index
        if self.use_detail_eval: # for per-sequence result
            self.occpano4d_metrics = OccPanoptic4DEvalDetail(
                                n_classes=len(self.occ_classes),
                                ignore=ignore)  
        else:
            self.occpano4d_metrics = OccPanoptic4DEval(
                                n_classes=len(self.occ_classes),
                                ignore=ignore)
        # reorganize the occ_results
        occ_results = list(self.flatten(occ_results))
        occ_results = sorted(occ_results, key=lambda x: x['sample_idx'])#? YzL
        seq_sem_pred = []
        seq_sem_gt = []
        seq_inst_pred = []
        seq_inst_gt = []
        seq_interest_mask = []

        print("\nStarting Evaluation...")
        processed_seq_set = set()
        seq_num = max(self.flag) + 1 # check if this is
        last_seq = -1
        with tqdm(total=seq_num) as pbar:
            for occ_pred_dict in occ_results:
                semantics_pred = occ_pred_dict["pano_sem"].reshape((200, 200, 16))
                instances_pred = occ_pred_dict["pano_inst"].reshape((200, 200, 16))
                curr_index = occ_pred_dict["index"]
                sample_idx = occ_pred_dict["sample_idx"]
                curr_seq = sample_idx % 1000000 // 1000
                if curr_seq == last_seq or last_seq == -1:
                    if curr_seq in processed_seq_set:
                        continue
                    seq_sem_gt, seq_inst_gt, seq_interest_mask = self.append_gt_info(
                            curr_index, 
                            seq_sem_gt, 
                            seq_inst_gt, 
                            seq_interest_mask)

                    seq_sem_pred.append(semantics_pred)
                    seq_inst_pred.append(instances_pred)

                    last_seq = curr_seq
                
                elif last_seq != curr_seq:
                    if last_seq in processed_seq_set:
                        continue
                    # add_batch
                    seq_sem_pred = np.stack(seq_sem_pred)
                    seq_sem_gt = np.stack(seq_sem_gt)
                    seq_inst_pred = np.stack(seq_inst_pred)
                    seq_inst_gt = np.stack(seq_inst_gt)
                    all_class_list = [1, 2, 3, 4]
                    for i in all_class_list:
                        if i not in self.inst_class_ids:
                            seq_inst_gt[seq_sem_gt == i] = 0
                    seq_interest_mask = np.stack(seq_interest_mask)
                    self.occpano4d_metrics.addBatch(
                        last_seq,
                        seq_sem_pred,
                        seq_inst_pred,
                        seq_sem_gt,
                        seq_inst_gt,
                        seq_interest_mask)
                    processed_seq_set.add(last_seq)
                    pbar.update(1)
                    # add info
                    seq_sem_pred = []
                    seq_sem_gt = []
                    seq_inst_pred = []
                    seq_inst_gt = []
                    seq_interest_mask = []

                    seq_sem_gt, seq_inst_gt, seq_interest_mask = self.append_gt_info(
                            curr_index, 
                            seq_sem_gt, 
                            seq_inst_gt, 
                            seq_interest_mask)

                    seq_sem_pred.append(semantics_pred)
                    seq_inst_pred.append(instances_pred)

                    last_seq = curr_seq
            # the latest seq add_batch
            if len(seq_sem_pred) > 0:
                seq_sem_pred = np.stack(seq_sem_pred)
                seq_sem_gt = np.stack(seq_sem_gt)
                seq_inst_pred = np.stack(seq_inst_pred)
                seq_inst_gt = np.stack(seq_inst_gt)
                # remove the id that don't need to track
                all_class_list = [1, 2, 3, 4]
                for i in all_class_list:
                    if i not in self.inst_class_ids:
                        seq_inst_gt[seq_sem_gt == i] = 0
                seq_interest_mask = np.stack(seq_interest_mask)
                self.occpano4d_metrics.addBatch(
                    last_seq,
                    seq_sem_pred,
                    seq_inst_pred,
                    seq_sem_gt,
                    seq_inst_gt,
                    seq_interest_mask)
                processed_seq_set.add(last_seq)
                pbar.update(1)
        if self.use_detail_eval:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            statistic_file = os.path.join(save_path, model_name + ".csv")
            self.occpano4d_metrics.save_statistic(statistic_file)
        
        res = {}
        res["panoptic4d"] = self.occpano4d_metrics.getPQ4D()
        return res

    def append_gt_info(self, index, seq_sem_gt, seq_inst_gt, seq_interest_mask):
        info = self.data_infos[index]
        pts_filename = info["lidar_points"]["lidar_path"].split("/")[-1]
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]

        split_set = "validation" if "train" not in self.ann_file else "training"
        occ_path = os.path.join(
            self.occupancy_path, split_set, seq_name, "{}_04.npz".format(frame_name))
        print(occ_path) 
        occ_gt = np.load(occ_path)

        semantics_gt = occ_gt["semantics"]
        instances_gt = occ_gt["instances"]
        
        mask_camera = occ_gt["mask_camera"].astype(bool) & occ_gt["infov"].astype(bool)
        #mask_camera = occ_gt["infov"].astype(bool)#YzL
        seq_sem_gt.append(semantics_gt)
        seq_inst_gt.append(instances_gt)
        seq_interest_mask.append(mask_camera)
        return seq_sem_gt, seq_inst_gt, seq_interest_mask

    def format_results(self, occ_results, savefile_prefix, **kwargs):
        if savefile_prefix is not None:
            mmcv.mkdir_or_exist(savefile_prefix)

        for occ_pred_dict in tqdm(occ_results):
            semantics_pred = occ_pred_dict["pano_sem"].reshape((200, 200, 16))
            instances_pred = occ_pred_dict["pano_inst"].reshape((200, 200, 16))
            sample_idx = occ_pred_dict["sample_idx"]
            save_path = os.path.join(savefile_prefix, "{}.npz".format(sample_idx))
            np.savez_compressed(save_path, pano_sem=semantics_pred, pano_inst=instances_pred)
        print("\nFinished.")