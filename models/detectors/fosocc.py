import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
#from models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from models.utils.structures import Instances
from models.utils.qim import build_qim
from models.utils.memory_bank import build_memory_bank
from copy import deepcopy
from mmdet.models.utils.transformer import inverse_sigmoid


def gen_dx_bx(xbound, ybound, zbound, device='cuda'):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).to(device)
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]]).to(device)
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).to(device)
    return dx, bx, nx

# this class is modified from MOTR
class RuntimeTrackerBase(object):
    # code from https://github.com/megvii-model/MOTR/blob/main/models/motr.py#L303
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    track_instances.obj_idxes[i] = -1

    def update_fix_label(self, track_instances: Instances, old_class_scores):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                # sleep time ++
                track_instances.disappear_time[i] += 1
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    track_instances.obj_idxes[i] = -1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] >= self.filter_score_thresh:
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]


@DETECTORS.register_module()
class FoSOcc(MVXTwoStageDetector):
    def __init__(self,
                 embed_dims=256, # embed_dims = 256, need to be fixed
                 num_query=300,
                 num_classes=16,
                 inst_class_ids=[1, 2, 3, 4],
                 grid_config={
                    'x': [-40, 40, 0.4],
                    'y': [-40, 40, 0.4],
                    'z': [-1, 5.4, 0.4],
                    'depth': [2.0, 42.0, 0.5]},
                 qim_args=dict(
                     qim_type='QIMBase',
                     merger_dropout=0, update_query_pos=False,
                     fp_ratio=0.3, random_drop=0.1),
                 mem_cfg=dict(
                     memory_bank_type='MemoryBank',
                     memory_bank_score_thresh=0.0,
                     memory_bank_len=4,
                 ),
                 time_interval=0.5,
                 with_velo=True,
                 score_threshold=0.3,
                 overlap_threshold=0.7,
                 filter_score_threshold=0.25, 
                 miss_tolerance=3,
                 img_view_transformer=None, 
                 volume_encoder_backbone=None,
                 volume_encoder_neck=None, 
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 loss_cfg=None,
                **kwargs):
        super(FoSOcc, self).__init__(**kwargs)
        '''
        depth_estimater_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }       
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        depth_estimater = DepthAnythingV2(**depth_estimater_configs[encoder])
        depth_estimater.load_state_dict(torch.load(f'/root/autodl-tmp/trackocc/TrackOcc/models/DepthAnythingV2/ckpt/depth_anything_v2_vits.pth', map_location='cpu'))# need to be fixed
        self.depth_estimater = depth_estimater.to('cuda').eval()
        '''
        if img_view_transformer:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        if volume_encoder_backbone:
            self.volume_encoder_backbone = builder.build_backbone(volume_encoder_backbone)
        if volume_encoder_neck:
            self.volume_encoder_neck = builder.build_neck(volume_encoder_neck)

        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.inst_class_ids = inst_class_ids
        self.grid_config = grid_config
        self.dx, self.bx, self.nx = gen_dx_bx(grid_config['x'],
                               grid_config['y'],
                               grid_config['z'])
        self.occ_size = [int((row[1] - row[0]) / row[2]) for row in [grid_config['x'], grid_config['y'], grid_config['z']]]
        self.time_interval = time_interval
        self.with_velo = with_velo

        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold
        #self.embed_dims = 256 # here it should be 256, but it may be 128 sometimes.
        #embed_dims = 256 # here it should be 256, but it may be 128 sometimes.
        self.reference_points = nn.Linear(self.embed_dims, 3)
        
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)
        
        self.mem_bank_len = mem_cfg['memory_bank_len']
        self.memory_bank = None
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_threshold,
            filter_score_thresh=filter_score_threshold,
            miss_tolerance=miss_tolerance) # hyper-param for removing inactive queries

        self.query_interact = build_qim(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.memory_bank = build_memory_bank(
            args=mem_cfg,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            )
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length

        self.criterion = builder.build_loss(loss_cfg)

        self.test_track_instances = None
        self.fp16_enabled = False       


    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('x'), out_fp32=True)
    def volume_encoder(self, x):
        x = self.volume_encoder_backbone(x)#CustomResNet3D
        compact_occ, volume_feats = self.volume_encoder_neck(x)#LSS
        
        #YzL
        
        return compact_occ, volume_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        B, N, C, imH, imW = img.shape
        img = img.view(B * N, C, imH, imW)
      
        x = self.img_backbone(img)
       
        if self.with_img_neck:
            x = self.img_neck(x)

        img_feats = []
        for img_feat in x:
            _, c, h, w = img_feat.size()
            img_feats.append(img_feat.view(B, N, c, h, w))

        return img_feats
    
    def extract_feat(self, img, img_metas=None):
        """Extract features from images and points."""
        
        
        cam_params = img[1:8] # rot, tran, intrin, cal, post_rot, post_tran, bda # here I may mask the cal
        #cam_params = cam_params[:3] + cam_params[4:]
        img_feats = self.extract_img_feat(img[0]) #

        mlp_input = self.img_view_transformer.get_mlp_input(*cam_params)
        # volume_feats only have one level(using LSS[YzL])
        
        volume_feats, depth= self.img_view_transformer([img_feats[2]] + img[1:8] + [mlp_input]) # 
        '''
        if img[0].shape[1] == 1: # need to be fixed
            with torch.no_grad():
                depth_input = img[0][:,0,:,:,:].permute(3, 2, 1, 0)
                depth_input = depth_input[..., -1].cpu().numpy()
                depth = self.depth_estimater.infer_image(depth_input)
                depth = depth.unsqueeze(0)
        '''
        
        # in 3D space
        #print(self.volume_encoder)
        #print(volume_feats.shape)
        compact_occ, volume_feats = self.volume_encoder(volume_feats) # x, (old_x_8, old_x_16, x_32)

        return depth, img_feats, compact_occ, volume_feats

    def _generate_empty_tracks(self, past_neg_pos=None):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        if past_neg_pos is not None:
            value = past_neg_pos.unsqueeze(0)
            new_pos = self.pos_attentions(query[..., :dim // 2].unsqueeze(0), value, value)[0].squeeze(0)
            new_pos = query[..., :dim // 2] + new_pos
        else:
            new_pos = query[..., :dim // 2]
        
        track_instances.ref_pts = self.reference_points(new_pos)
        track_instances.query = torch.cat([new_pos, query[..., dim // 2:]], dim=-1)
        occ_size_flatten = np.prod(self.occ_size).item()
        

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device)

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        # x, y, z (vx, vy, vz)
        if not self.with_velo:
            pred_centers_init = torch.zeros(
                (len(track_instances), 3), dtype=torch.float, device=device)
        else:
            pred_centers_init = torch.zeros(
                (len(track_instances), 6), dtype=torch.float, device=device)
        
        track_instances.pred_centers = pred_centers_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)
        
        track_instances.pred_masks = torch.zeros(
            (len(track_instances), occ_size_flatten),
            dtype=torch.float, device=device)
        
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len),
            dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros(
            (len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embedding.weight.device)

    def _copy_tracks_for_loss(self, tgt_instances):

        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_centers = torch.zeros(
            (len(track_instances), 3), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes),
            dtype=torch.float, device=device)
        track_instances.pred_masks = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)

        track_instances.save_period = deepcopy(tgt_instances.save_period)
        return track_instances.to(self.query_embedding.weight.device)

    @force_fp32()
    def forward_train(self, 
                      img_metas=None, 
                      img_inputs=None, 
                      gt_depth=None, 
                      voxel_semantics=None, 
                      voxel_instances=None, 
                      instid_and_clsid=None, 
                      visible_mask=None, 
                      infov_mask=None, 
                      gt_offsets = None, 
                      **kwargs):

        bs = img_inputs[0][0].size(0)
        num_frame = len(img_inputs)
        device = img_inputs[0][0].device
        track_instances = self._generate_empty_tracks()#For the first frame (i = 1), track query assignment ω1 tr is an empty set

        gt_instances_list = []
        curr2next_ego_list = []
        img_metas_single_list = []
        for i in range(num_frame):
            # init metas single
            img_metas_single = dict()
            for key in img_metas[0]:
                img_metas_single[key] = img_metas[0][key][i]
                if key == 'ego2global':
                    curr_ego2global = torch.as_tensor(img_metas[0][key][i], device=device)
                    next_ego2global = img_metas[0][key][i+1] if i != num_frame - 1 else img_metas[0][key][i]
                    next_ego2global = torch.as_tensor(next_ego2global, device=device)
                    curr2next_ego = torch.inverse(next_ego2global) @ curr_ego2global
                    curr2next_ego_list.append(curr2next_ego)
            img_metas_single_list.append(img_metas_single)
            # init gt instances!
            gt_instances = Instances((1, 1))
            inst_ids = instid_and_clsid[i][0][:, 0]
            labels = instid_and_clsid[i][0][:, 1]
            voxel_instances_i = voxel_instances[i][0]
            thing_masks = []
            thing_centers = []
            if len(inst_ids) > 0:
                for id in inst_ids:
                    thing_mask = (voxel_instances_i == id)

                    # get the 3D center of the mask
                    temp1 = thing_mask.nonzero().float()
                    thing_center = torch.mean(thing_mask.nonzero().float(), dim=0)
                    # transform to phsical length using self.dx, self.bx, self.nx
                    thing_center = thing_center * self.dx + self.bx
                    if self.with_velo:
                        if i != num_frame - 1:
                            next_frame_mask = (voxel_instances[i + 1][0]  == id)
                            if next_frame_mask.sum() > 0:

                                next_frame_center = next_frame_mask.nonzero().float().mean(dim=0)
                                next_frame_center = (next_frame_center * self.dx + self.bx)
                                next2curr_ego = torch.inverse(curr2next_ego)
                                next_frame_center_in_curr = next2curr_ego[:3, :3].to(torch.float32) @ next_frame_center.to(torch.float32) + next2curr_ego[:3, 3].to(torch.float32)#YzL
                                velocity = (next_frame_center_in_curr - thing_center) / self.time_interval
                            else:
                                # create a nan tensor
                                velocity = torch.full((3,), float('nan'), dtype=torch.float32, device=device)
                        else:
                            velocity = torch.full((3,), float('nan'), dtype=torch.float32, device=device)
                        thing_center = torch.cat([thing_center, velocity], dim=0)
                    
                    thing_masks.append(thing_mask)
                    thing_centers.append(thing_center)
                thing_masks = torch.stack(thing_masks, dim=0).flatten(1)
                thing_centers = torch.stack(thing_centers, dim=0)
            else:
                thing_masks = torch.zeros((0, voxel_instances_i.size(0)), dtype=torch.float32, device=voxel_instances_i.device)
                if not self.with_velo:
                    thing_centers = torch.zeros((0, 3), dtype=torch.float32, device=voxel_instances_i.device)
                else:
                    thing_centers = torch.zeros((0, 6), dtype=torch.float32, device=voxel_instances_i.device)
            
            gt_instances.obj_ids = inst_ids
            gt_instances.labels = labels
            gt_instances.masks = thing_masks
            gt_instances.centers = thing_centers
            gt_instances_list.append(gt_instances)
        losses = {}

        self.criterion.initialize_for_single_clip(gt_instances_list)
        #print(num_frame)#YzL
        for i in range(num_frame): 
            img_metas_single = [img_metas_single_list[i]]
            frame_res, not_track_loss = self._forward_single(track_instances, img_inputs[i], curr2next_ego_list[i], gt_depth[i], voxel_semantics[i], gt_offsets[i],\
                                  i, visible_mask[i], infov_mask[i], img_metas=img_metas_single)
            track_instances = frame_res['track_instances']
            
            losses.update(not_track_loss)

        losses.update(not_track_loss)
        track_loss = self.criterion.losses_dict
        losses.update(track_loss)
        return losses
    
    def _forward_single(self,
                        track_instances, 
                        img_inputs, 
                        curr2next_ego, 
                        gt_depth, 
                        voxel_semantics, 
                        gt_offsets, 
                        frame_idx=0, 
                        visible_mask=None, 
                        infov_mask=None, 
                        img_metas=None):
        depth, img_feats, compact_occ, volume_feats = self.extract_feat(img=img_inputs, img_metas=img_metas)

        not_track_dict, cls_pred_list, mask_pred_list, ref_pts_and_velo_list,\
            query_feats, last_ref_points = self.pts_bbox_head(track_instances.query,
                                                                    track_instances.ref_pts,
                                                                    img_feats, 
                                                                    compact_occ, 
                                                                    volume_feats,
                                                                    img_inputs[1:8],
                                                                    infov_mask, 
                                                                    img_metas
                                                                    )# here I may introduce the gt_offsets
        # not track loss
        not_track_losses = {}
        
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        not_track_dict['gt_offsets'] = gt_offsets
        not_track_dict['voxel_semantics'] = voxel_semantics
        not_track_dict['visible_mask'] = visible_mask
        losses_pts = self.pts_bbox_head.loss(**not_track_dict)
        #loss_offsets = 
        not_track_losses.update(loss_depth=loss_depth)
        not_track_losses.update(losses_pts)
        not_track_losses = {'frame_{}_{}'.format(frame_idx, key): value for key, value in not_track_losses.items()}

        with torch.no_grad():
            track_scores = cls_pred_list[-1][0, :, self.inst_class_ids].sigmoid().max(dim=-1).values   
        
        nb_dec = len(cls_pred_list)
        #print(nb_dec)
        # the track id will be assigned by the matcher.
        track_instances_list = [self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)]
        track_instances.output_embedding = query_feats[0] 

        ref_pts = self.ref_pts_align(last_ref_points[0], ref_pts_and_velo_list[-1][0], curr2next_ego)

        track_instances.ref_pts = ref_pts
        track_instances_list.append(track_instances)
        for i in range(nb_dec):
            out = {}
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = cls_pred_list[i][0]
            track_instances.pred_centers = ref_pts_and_velo_list[i][0]
            track_instances.pred_masks = mask_pred_list[i][0].flatten(1)

            out['track_instances'] = track_instances
            mask_camera = visible_mask[0].flatten()
            track_instances = self.criterion.match_for_single_frame(
                out, mask_camera, i, if_step=(i == (nb_dec - 1)))

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances) # attention with history embedding

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)
        out['track_instances'] = out_track_instances

        return out, not_track_losses

    def ref_pts_align(self, ref_pts, ref_pts_and_velo, curr2next_ego):
        aligned_ref_pts = ref_pts.sigmoid().clone()

        pc_range = self.pc_range
        aligned_ref_pts[..., 0:1] = aligned_ref_pts[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        aligned_ref_pts[..., 1:2] = aligned_ref_pts[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        aligned_ref_pts[..., 2:3] = aligned_ref_pts[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
        
        if ref_pts_and_velo.shape[-1] == 6:
            aligned_ref_pts = aligned_ref_pts[:, :3] + ref_pts_and_velo[:, 3:6] * self.time_interval
        padding = aligned_ref_pts.new_ones(aligned_ref_pts.size(0), 1)
        aligned_ref_pts = torch.cat([aligned_ref_pts, padding], dim=1)
        aligned_ref_pts = curr2next_ego[None, ...].to(torch.float32) @ aligned_ref_pts.unsqueeze(-1) # May need fixeds
        aligned_ref_pts = aligned_ref_pts.squeeze(-1)[:, :3]
        #normalization(YzL)
        aligned_ref_pts[..., 0:1] = (aligned_ref_pts[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        aligned_ref_pts[..., 1:2] = (aligned_ref_pts[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        aligned_ref_pts[..., 2:3] = (aligned_ref_pts[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        aligned_ref_pts = inverse_sigmoid(aligned_ref_pts)

        return aligned_ref_pts

    def forward_test(self, img_metas, img_inputs=None, infov_mask=None, **kwargs):

        assert type(img_metas[0]['index']) == int, "index's type is int meaning num_frame=1"
        
        if img_metas[0]['start_of_sequence']:
            track_instances = self._generate_empty_tracks()
            self.track_base.clear()
        else:
            track_instances = self.test_track_instances
        
        prev2curr_ego = torch.inverse(img_metas[0]['curr_to_prev_ego_rt'])\
                            .to(img_inputs[0].device)
        frame_res = self._inference_single(track_instances, img_inputs, 
                        prev2curr_ego, infov_mask, img_metas=img_metas)
        
        track_instances = frame_res['track_instances']

        active_instances = self.query_interact._select_active_tracks(
            dict(track_instances=track_instances))
        
        self.test_track_instances = track_instances

        out_dict = self.merge_panoseg(active_instances, frame_res['stuff_pred_logits'], frame_res['stuff_pred_masks'], img_metas)

        curr_inst = out_dict['pano_inst'].cpu().numpy().astype(np.int16)
        curr_sem = out_dict['pano_sem'].cpu().numpy().astype(np.uint8)

        out_dict = {
            'index':img_metas[0]['index'],
            'sample_idx':img_metas[0]['sample_idx'],
            'pano_inst': curr_inst,
            'pano_sem': curr_sem, 
        }
        output_list = []
        output_list.append(out_dict)

        return output_list


    def _inference_single(self,
                        track_instances, 
                        img_inputs, 
                        prev2curr_ego, 
                        infov_mask=None, 
                        img_metas=None):
        
        # align active inst ref_pts to current frame
        active_inst = track_instances[track_instances.obj_idxes > 0]
        other_inst = track_instances[track_instances.obj_idxes <= 0]

        if len(active_inst) > 0:
            ref_pts = active_inst.ref_pts
            ref_pts_and_velo = active_inst.pred_centers
            ref_pts = self.ref_pts_align(ref_pts, ref_pts_and_velo, prev2curr_ego)
            active_inst.ref_pts = ref_pts
            track_instances = Instances.cat([other_inst, active_inst])

        depth, img_feats, compact_occ, volume_feats = self.extract_feat(img=img_inputs, img_metas=img_metas)
        not_track_dict, cls_pred_list, mask_pred_list,ref_pts_and_velo_list,\
            query_feats, last_ref_points = self.pts_bbox_head(track_instances.query,
                                                            track_instances.ref_pts,
                                                            img_feats, 
                                                            compact_occ, 
                                                            volume_feats,
                                                            img_inputs[1:8],
                                                            infov_mask, 
                                                            img_metas)
        
        out = {'seg_pred': not_track_dict['seg_pred_list'][-1] if len(not_track_dict['seg_pred_list']) > 0 else None,
               'stuff_pred_logits':not_track_dict['stuff_cls_pred_list'][-1],
               'stuff_pred_masks':not_track_dict['stuff_mask_pred_list'][-1],
               'pred_logits': cls_pred_list[-1],
               'pred_masks': mask_pred_list[-1],
               'ref_pts': last_ref_points}

        track_scores = cls_pred_list[-1][0, :, self.inst_class_ids].sigmoid().max(dim=-1).values   

        track_instances.scores = track_scores
        track_instances.pred_logits = cls_pred_list[-1][0]  # [300, num_cls]
        track_instances.pred_centers = ref_pts_and_velo_list[-1][0]
        track_instances.pred_masks = mask_pred_list[-1][0].flatten(1)  # [300, num_voxel]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_points[0]

        self.track_base.update(track_instances)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        out_track_instances = self.query_interact(tmp)
        out['track_instances'] = out_track_instances

        return out

    def merge_panoseg(self, active_instances, stuff_logits, stuff_masks, img_metas):
        active_idxes = (active_instances.scores >= self.track_base.filter_score_thresh)
        active_instances = active_instances[active_idxes]

        thing_mask_cls = active_instances.pred_logits.sigmoid()
        thing_mask_pred = active_instances.pred_masks.sigmoid()
        obj_idxes = active_instances.obj_idxes + 1 # 0-based to 1-based, in result, 0 is for stuff

        stuff_mask_cls = stuff_logits[0].sigmoid()
        stuff_mask_pred = stuff_masks[0].flatten(1).sigmoid()
        stuff_obj_idxes = torch.full((stuff_mask_cls.size(0),), 0, dtype=torch.long, device=stuff_mask_cls.device)

        mask_cls = torch.cat([thing_mask_cls, stuff_mask_cls], dim=0)
        mask_pred = torch.cat([thing_mask_pred, stuff_mask_pred], dim=0)
        obj_idxes = torch.cat([obj_idxes, stuff_obj_idxes], dim=0)
        out_dict = {}
        pano_inst, pano_sem = self.merge_panoseg_single(mask_cls, mask_pred, obj_idxes)

        out_dict['pano_inst'] = pano_inst # [N]
        out_dict['pano_sem'] = pano_sem

        return out_dict

    def merge_panoseg_single(self, mask_cls, mask_pred, obj_idxes):
        occ_x, occ_y, occ_z = self.occ_size
        scores, labels = mask_cls.max(-1) 
        # super-param
        num_classes = self.num_classes

        # filter out low score and background instances
        if num_classes == 19: #Kitti360 ,may need fixed
            keep = labels.ne(num_classes - 1) & (scores > self.score_threshold)
            #keep = labels.ne(num_classes - 1) & (scores > self.score_threshold)
        elif num_classes == 16:
            keep = labels.ne(num_classes - 1) & (scores > self.score_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep] # (filter_q, n)
        cur_obj_idxes = obj_idxes[keep]

        cur_prob_masks = cur_scores.view(-1, 1) * cur_masks

        N = cur_masks.shape[-1]
        instance_seg = torch.zeros((N), dtype=torch.int32, device=cur_masks.device)
        semantic_seg = torch.ones((N), dtype=torch.int32, device=cur_masks.device) * (num_classes - 1)
        
        # skip all process if no mask is detected
        if cur_masks.shape[0] != 0:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # [N]: max_pro q indice
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                current_segment_id = cur_obj_idxes[k].item()
                # objects are treated as instances
                # is_thing = pred_class in inst_class_ids

                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item() # w/o semantic and instance meaning
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    instance_seg[mask] = current_segment_id
                    semantic_seg[mask] = pred_class
                    
        instance_seg = instance_seg
        semantic_seg = semantic_seg
        
        return instance_seg, semantic_seg   # B, N