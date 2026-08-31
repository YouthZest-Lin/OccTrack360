import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import os.path as osp
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from copy import deepcopy
import cv2
import os
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
import torch.nn as nn
from mmdet.datasets.pipelines import to_tensor, DefaultFormatBundle
from models.utils.VoxNT_Trick import *
def mmlabNormalize(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                   to_rgb=True, debug=False):
    from mmcv.image.photometric import imnormalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    to_rgb = to_rgb
    if debug:
        print('warning, debug in mmlabNormalize')
        img = np.asarray(img) # not normalize for visualization
    else:
        img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

@PIPELINES.register_module()
class DefaultFormatBundle3DTrack(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3DTrack, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            points_cat = []
            for point in results['points']:
                assert isinstance(point, BasePoints)
                points_cat.append(point.tensor)
            # results['points'] = DC(torch.stack(points_cat, dim=0))
            results['points'] = DC(points_cat)

        # if 'img_inputs' in results:
        #     imgs_list = results['img_inputs']
        #     results['img_inputs'] = DC(imgs_list)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3DTrack, self).__call__(results)
        return results

@PIPELINES.register_module()
class OccWaymoPrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',

        normalize_cfg=dict(
             mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False
        )
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.normalize_cfg = normalize_cfg

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # post_rot *= resize # （W, H）
        post_rot[0, 0] *= resize[0]
        post_rot[1, 1] *= resize[1]
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize_w = float(fW) / float(W)
            resize_w += np.random.uniform(*self.data_config['resize'])
            resize_h = float(fH) / float(H)
            resize_h += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize_w), int(H * resize_h))
            resize = [resize_w, resize_h]

            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH # self.data_config['crop_h'] = 0 -> crop_h = newH - fH 
            crop_w = int(np.random.uniform(0, max(0, newW - fW))) 
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH) # 这里已经是fH, fW长度的形状了，只是要crop
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize_w = float(fW) / float(W)
            resize_h = float(fH) / float(H)
            resize_w += self.data_config.get('resize_test', 0.0)
            resize_h += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
                resize_dims = (int(W * resize), int(H * resize))
                resize = [resize, resize]
            else:
                resize_dims = (int(W * resize_w), int(H * resize_h))
                resize = [resize_w, resize_h]

            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


    def get_inputs(self, results, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        sensor2egos = []
        ego2globals = []
        cam_names = self.choose_cams() # list of cam_names
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        results['img_augs'] = {}
        for cam_name in cam_names:
            cam_data = results['curr']['images'][cam_name]
            filename = cam_data['img_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam2img'])[:3,:3] # (3, 4)
            # in the form of waymo in mmdet3d, lidar means ego 
            sensor2ego = torch.Tensor(np.linalg.inv(cam_data['lidar2cam']))
            rot = sensor2ego[:3, :3]
            tran = sensor2ego[:3, 3]
            ego2global = torch.Tensor(results['curr']['ego2global'])
            # image view augmentation (resize, crop, horizontal flip, rotate)
            if results.get('tta_config', None) is not None:
                flip = results['tta_config']['tta_flip']
            else: flip = None
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            results['img_augs'][cam_name] = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img, **self.normalize_cfg))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)

        imgs = torch.stack(imgs)
        
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)

        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        return (imgs, rots, trans, intrins, post_rots, post_trans), (sensor2egos, ego2globals)

    def __call__(self, results):
        results['img_inputs'], results['aux_cam_params'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class OccWaymoLoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, boxes_before_bda_aug=True, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.boxes_before_bda_aug = boxes_before_bda_aug

    def sample_bda_augmentation(self, tta_config=None):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            if tta_config is not None:
                flip_dx = tta_config['flip_dx']
                flip_dy = tta_config['flip_dy']
            else:
                flip_dx = False
                flip_dy = False

        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if not self.boxes_before_bda_aug and gt_boxes.shape[0] > 0:
                gt_boxes[:, :3] = (
                    rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
                gt_boxes[:, 3:6] *= scale_ratio
                gt_boxes[:, 6] += rotate_angle
                if flip_dx:
                    gt_boxes[:,
                            6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                            6]
                if flip_dy:
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
                gt_boxes[:, 7:] = (
                    rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        # NOTE: we don't need bbox. if you want bboxes, please fix the code
        if 'gt_bboxes_3d' in results:
            gt_boxes = results['gt_bboxes_3d'].tensor
        else:
            gt_boxes = torch.zeros(0, 9)
        if 'gt_labels_3d' not in results:
            results['gt_labels_3d'] = torch.zeros(0)

        gt_boxes[:,2] = gt_boxes[:,2] + 0.5*gt_boxes[:,5]
        tta_confg = results.get('tta_config', None)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(tta_confg
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        return results

   
@PIPELINES.register_module()
class OccWaymoLoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 dtype='float32',
                 file_client_args=dict(backend='disk'),
                 translate2ego=True,
                 ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        if dtype=='float32':
            self.dtype = np.float32
        elif dtype== 'float16':
            self.dtype = np.float16
        else:
            assert False
        self.translate2ego = translate2ego

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=self.dtype)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=self.dtype)

        return points


    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]



        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class OccWaymoPointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]


        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]


        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
     
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]

            lidar2cam = results['curr']['images'][cam_name]['lidar2cam']
            lidar2cam = torch.tensor(lidar2cam)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])  
            depth_map_list.append(depth_map)
            
            # from torchvision.utils import save_image
            # save_image(imgs[[cid]], './img{}.png'.format(cid))

            # # draw the depth_map and save as a png
            # import matplotlib.pyplot as plt
            # plt.imshow(depth_map.cpu().numpy(), cmap='plasma')
            # plt.colorbar()  # 添加色彩条
            # plt.savefig('./depth_map{}.png'.format(cid))
            # plt.close()

        depth_map = torch.stack(depth_map_list)

        results['gt_depth'] = depth_map

        return results

@PIPELINES.register_module()
class OccWaymoLoadOccConsistant(object):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __init__(self, occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                    num_classes=16,
                    mask='mask_camera',
                    panoptic=True,
                    inst_class_ids=[1,2,3,4],
                    larger_resolute=False):
        self.occupancy_path = occupancy_path
        self.num_classes = num_classes
        self.mask = mask
        self.panoptic = panoptic
        self.inst_class_ids = inst_class_ids
        self.larger_resolute = larger_resolute


    
    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        
        def cal_focus_feature(semantics, instances):
            gt_offsets = compute_all_direction_distances(semantics + semantics.max() * instances) 


            gt_inst_offsets = torch.zeros_like(gt_offsets)

            gt_focus_feature = torch.zeros((1,200,200,16))

            for inst in torch.unique(instances):

                if inst == 0:
                    continue

                gt_inst_offsets_per_inst = gt_offsets * (
                            instances == inst).to(
                    torch.float32)
                for i in range(6):
                    gt_inst_offsets_per_inst[i] = gt_inst_offsets_per_inst[i] / (gt_inst_offsets_per_inst[i].max() + 1e-8) 


                gt_inst_offsets = gt_inst_offsets + gt_inst_offsets_per_inst
                gt_focus_feature_per_inst = torch.prod(gt_inst_offsets_per_inst,dim = 0) / (torch.prod(gt_inst_offsets_per_inst,dim = 0).max() + 1e-8)
                gt_focus_feature += gt_focus_feature_per_inst
            return gt_focus_feature, gt_inst_offsets
        
        pts_filename = results['pts_filename'].split('/')[-1]
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.larger_resolute:
            file_path=os.path.join(self.occupancy_path, seq_name,  '{}.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.occupancy_path, seq_name, '{}_04.npz'.format(frame_name))

        data = np.load(file_path)

        assert self.mask is not None, 'mask must be specified'
        if self.mask == 'mask_camera':
            visible_mask = torch.tensor(data['mask_camera']).to(torch.bool)
        else:
            visible_mask = torch.tensor(data['mask_lidar']).to(torch.bool)
            
        infov_mask = torch.tensor(data['infov']).to(torch.bool)
        visible_mask = torch.logical_and(visible_mask, infov_mask)
        
        semantics = torch.tensor(data['semantics'])        
        mask_semantics = semantics[visible_mask]

        instances =  torch.tensor(data['instances'])
        
        # remove the id that don't need to track
        all_class_list = [1, 2, 3, 4]
        for i in all_class_list:
            if i not in self.inst_class_ids:
                instances[semantics == i] = 0

        mask_instances = instances[visible_mask]
        instid_and_clsid = []
        unique_inst_ids = torch.unique(mask_instances.flatten(-1))

        for i in unique_inst_ids:
            if i == 0:
                continue
            class_id = torch.unique(mask_semantics[mask_instances == i])
            if len(class_id) == 0:
                continue

            assert class_id.shape[0] == 1, "each instance must belong to only one class"
            id_and_clsid = [i.item(), class_id[0].item()]

            instid_and_clsid.append(id_and_clsid)
        
        if len(instid_and_clsid) == 0:
            instid_and_clsid = torch.zeros([0, 2], dtype=torch.int64)

        if results['rotate_bda'] != 0: # nearest neighbor interpolation
            semantics = semantics.permute(2, 1, 0) # (X, Y, Z) -> (Z, Y, X)
            semantics = rotate(semantics, -results['rotate_bda'], fill=self.num_classes - 1).permute(2, 1, 0) # (Z, Y, X) -> (X, Y, Z)
            
            visible_mask = visible_mask.permute(2, 1, 0)
            visible_mask = rotate(visible_mask, -results['rotate_bda'], fill=False).permute(2, 1, 0)
            
            infov_mask = infov_mask.permute(2, 1, 0)
            infov_mask = rotate(infov_mask, -results['rotate_bda'], fill=False).permute(2, 1, 0)

            instances = instances.permute(2, 1, 0)
            instances = rotate(instances, -results['rotate_bda'], fill=0).permute(2, 1, 0)
            
        if results['flip_dx']:
            semantics = torch.flip(semantics, [0])

            visible_mask = torch.flip(visible_mask, [0])
            infov_mask = torch.flip(infov_mask, [0])

            instances = torch.flip(instances, [0])

        if results['flip_dy']:
            semantics = torch.flip(semantics, [1])

            visible_mask = torch.flip(visible_mask, [1])
            infov_mask = torch.flip(infov_mask, [1])

            instances = torch.flip(instances, [1])

        results['visible_mask'] = visible_mask

        results['infov_mask'] = infov_mask
    
        results['voxel_semantics'] = semantics

        results['voxel_instances'] = instances

        results['instid_and_clsid'] = DC(to_tensor(instid_and_clsid))
        
        gt_focus_feature, gt_inst_offsets = cal_focus_feature(results['voxel_semantics'] , results['voxel_instances'])                


        results['gt_offsets'] = torch.cat((gt_inst_offsets, gt_focus_feature), dim = 0)
        return results