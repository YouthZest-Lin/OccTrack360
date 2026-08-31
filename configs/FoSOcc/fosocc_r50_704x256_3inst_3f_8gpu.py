num_gpus = 1
samples_per_gpu = 1
workers_per_gpu = 5
num_queries = 200
num_stuff_queries = 50

num_iters_per_epoch = int(31617 // (num_gpus * samples_per_gpu))
by_epoch = False
num_epochs = 24
checkpoint_epoch_interval = 12
use_custom_eval_hook = True

# for tracking
score_threshold = 0.3
overlap_threshold = 0.7
filter_score_threshold = 0.25
miss_tolerance = 3

# for localization loss
center_supervise = True
with_velo = True

num_decoder_layers = 1
with_reg_branches = True

semantic_loss_flag = True
sem_loss_weight = 2

ignore_index = [9] # motorcycle

# Dataset settings

work_dirs = "./work_dirs"

dataset_root = "data/TrackOcc-waymo/kitti_format/"
occupancy_path = "data/TrackOcc-waymo/pano_voxel04/"
train_occ_path = occupancy_path + "training/"
val_occ_path = occupancy_path + "validation/"
val_anno_file = "waymo_infos_val_jpg.pkl"
train_anno_file = "waymo_infos_train_jpg.pkl"

test_use_sequence_group_flag = True

filter_empty_gt = True

mask = "mask_camera"  # if don't plan use mask, set mask=None

num_feature_levels = 4
with_cp = False
sync_bn = True
instance_class_names = [
    'vehicle',
    'pedestrian',
    'cyclist',
]
inst_class_ids = [1, 2, 4]
occ_class_names = [
    'GO', 'vehicle', 'pedestrian', 'sign', 'cyclist',
    'trafficlight', 'pole', 'constructioncone', 
    'bicycle', 'motorcycle',
    'building', 
    'vegetation', 'treetrunk',
    'road', 'walkable', 'free'] # 16

class_weight_multiclass = [
    21.996729830048952,
    7.504469780801267, 10.597629961083673, 12.18107968968811, 15.143940258446506, 13.035521328502758, 
    9.861234292376812, 13.64431851057796, 15.121236434460473, 21.996729830048952, 6.201671013759701, 
    5.7420517938838325, 9.768712859518626, 3.4607400626606317, 4.152268220983671, 1.000000000000000,
]

num_classes = len(occ_class_names)

load_interval = 5

data_config = {
    'cams': [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
        'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT'
    ],
    'Ncams': 5,
    'input_size': (256, 704), 
    'src_size': (1280, 1920),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [2.0, 42.0, 0.5],
}

occformer_grid_config = {
    'x': [-40, 40, 1.6],
    'y': [-40, 40, 1.6],
    'z': [-1, 5.4, 0.4],
}

compact_occ_x = int((occformer_grid_config['x'][1] - occformer_grid_config['x'][0]) / occformer_grid_config['x'][2])
compact_occ_y = int((occformer_grid_config['y'][1] - occformer_grid_config['y'][0]) / occformer_grid_config['y'][2])
compact_occ_z = int((occformer_grid_config['z'][1] - occformer_grid_config['z'][0]) / occformer_grid_config['z'][2])


depth_categories = 80

# NOTE: our model design cannot use bda augmentation
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0,
    flip_dy_ratio=0,
    )

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

numC_Trans = 32
_dim_ = 256
_pos_dim_ = [96, 96, 64]
_ffn_dim_ = _dim_ * 2

model = dict(
    type='FoSOcc',
    num_classes=num_classes,
    num_query=num_queries,
    with_velo=with_velo,
    grid_config=grid_config,
    score_threshold=score_threshold,
    overlap_threshold=overlap_threshold,
    filter_score_threshold=filter_score_threshold,
    miss_tolerance=miss_tolerance,
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=num_classes,
        center_supervise=center_supervise,
        loss_center_weight=0.3,
        loss_cls_weight=2.0,
        loss_mask_weight=100.0, # 5
        loss_dice_weight=5.0,
    ),
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.3, random_drop=0.1), # hyper-param for query dropping
    mem_cfg=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=with_cp),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4,
        upsample_cfg=dict(mode='nearest'),),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=_dim_,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96, with_cp=with_cp),
        downsample=16),
    volume_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        num_layer=[1, 2, 4],
        num_channels=[numC_Trans, numC_Trans*2, numC_Trans*4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2],
        with_cp=with_cp,
        norm_cfg=dict(type='BN3d', requires_grad=True),),
    volume_encoder_neck=dict(
        type='LSSFPN3D',
        in_channels=numC_Trans*7,
        out_channels=numC_Trans,
        reverse=True,
        size=(50, 50, 16)), # (x, y, z)
    pts_bbox_head=dict(
        type='FoSOccHead',
        in_channels=numC_Trans,
        embed_dims=_dim_,
        num_stuff_queries=num_stuff_queries,
        num_classes=num_classes,
        inst_class_ids=inst_class_ids,
        with_velo=with_velo,
        num_decoder_layers=num_decoder_layers,
        with_reg_branches=with_reg_branches,
        class_weight_multiclass=class_weight_multiclass,
        semantic_loss_flag=semantic_loss_flag,
        sem_loss_weight=sem_loss_weight,
        transformer=dict( # enhance [50, 50, 16] occ features
            type='TransformerMSOcc',
            embed_dims=_dim_,
            num_feature_levels=num_feature_levels,
            num_cams=data_config['Ncams'],
            encoder=dict(
                type='OccEncoder_ori',
                num_layers=1,
                grid_config=occformer_grid_config,
                data_config=data_config,
                pc_range=point_cloud_range,
                return_intermediate=False,
                fix_bug=True,
                transformerlayers=dict(
                    type='OccFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention3D',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_points=4),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=num_feature_levels),
                            embed_dims=_dim_,)
                    ],
                    ffn_embed_dims=_dim_,
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        transformer_decoder=dict(
            type='MaskOccDecoder',
            return_intermediate=True,
            return_ref_points=True,
            num_layers=num_decoder_layers,
            transformerlayers=dict(
                type='MaskOccDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiScaleDeformableAttention3D',
                        embed_dims=_dim_,
                        num_levels=1,
                        num_points=4,),
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=2 * _dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                    'ffn', 'norm'))),
        positional_encoding=dict(
            type='CustomLearnedPositionalEncoding3D',
            num_feats=_pos_dim_,
            row_num_embed=int(compact_occ_x),
            col_num_embed=int(compact_occ_y),
            tub_num_embed=int(compact_occ_z)
        ),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2Former3DLoss',
                num_classes=len(occ_class_names),
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
        ),
    ),
)


# Data
dataset_type = 'WindowTrackOccWaymoDataset'
file_client_args = dict(backend='disk')

data_prefix=dict(
    pts='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_LEFT='training/image_1',
    CAM_FRONT_RIGHT='training/image_2',
    CAM_SIDE_LEFT='training/image_3',
    CAM_SIDE_RIGHT='training/image_4')

train_pipeline_post = [
    dict(type='DefaultFormatBundle3DTrack', class_names=instance_class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_depth', 'visible_mask', 'infov_mask', 
                                 'voxel_semantics', 'voxel_instances', 'instid_and_clsid', 'gt_offsets'],
                        meta_keys = ('index', 'sample_idx', 'ego2global', 'start_of_sequence', 
                            'waymo_get_rt_matrix', 'curr_to_prev_ego_rt', 'prev_ego_to_global_rt', 
                            'global_to_curr_ego_rt', 'box_mode_3d', 'box_type_3d', 'aux_cam_params'))
]

train_pipeline = [
    dict(
        type='OccWaymoPrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='OccWaymoLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf),
    dict(
        type='OccWaymoLoadPointsFromFile',
        coord_type='LIDAR', load_dim=6, use_dim=3,
        file_client_args=file_client_args),
    dict(type='OccWaymoPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='OccWaymoLoadOccConsistant', occupancy_path=train_occ_path, mask=mask, inst_class_ids=inst_class_ids),
]

test_pipeline = [
    dict(type='OccWaymoPrepareImageInputs', data_config=data_config),
    dict(
        type='OccWaymoLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='OccWaymoLoadOccConsistant',  occupancy_path=val_occ_path, mask=mask, inst_class_ids=inst_class_ids),
    dict(
        type='DefaultFormatBundle3D',
        class_names=instance_class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs', 'visible_mask', 'infov_mask',
                                'voxel_semantics', 'voxel_instances', 'instid_and_clsid'], 
                        meta_keys = ('index', 'sample_idx', 'ego2global', 'start_of_sequence', 
                                'waymo_get_rt_matrix', 'curr_to_prev_ego_rt', 'prev_ego_to_global_rt', 
                                'global_to_curr_ego_rt', 'box_mode_3d', 'box_type_3d'))
]

test_data_config = dict(
    type=dataset_type,
    data_root=dataset_root,
    ann_file=dataset_root + val_anno_file,
    occupancy_path=occupancy_path,
    data_prefix=data_prefix,
    pipeline=test_pipeline,
    img_info_prototype='bevdet',
    ignore_index=ignore_index,
    inst_class_ids=inst_class_ids,
    occ_classes=occ_class_names,
    modality=input_modality,
    test_mode=True,
    load_interval=load_interval,
    use_sequence_group_flag=test_use_sequence_group_flag,
    box_type_3d='LiDAR')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + train_anno_file,
        occupancy_path=occupancy_path,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        inst_class_ids=inst_class_ids,
        occ_lasses=occ_class_names,
        pipeline_post=train_pipeline_post,
        test_mode=False,
        modality=input_modality,
        img_info_prototype='bevdet',
        load_interval=load_interval,
        use_sequence_group_flag=False,
        filter_empty_gt=filter_empty_gt,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

dist_params = dict(backend='nccl')

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(by_epoch=by_epoch, interval=checkpoint_epoch_interval * num_iters_per_epoch)

# logging
log_level = 'INFO'
log_config = dict(
    interval= 50, #50,
    hooks=[
        dict(type='TextLoggerHook',
            by_epoch=by_epoch),
    ])

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=checkpoint_epoch_interval * num_iters_per_epoch,
    ),
]

# evaluation
evaluation = dict(interval= checkpoint_epoch_interval * num_iters_per_epoch, 
                  by_epoch=by_epoch, pipeline=test_pipeline)

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'

revise_keys = [('backbone', 'img_backbone')]

# fp16 setting
fp16 = dict(loss_scale='dynamic')
