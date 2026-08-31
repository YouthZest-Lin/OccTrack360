import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from typing import Tuple
from torch import Tensor
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import build_loss
from models.utils.matcher import HungarianMatcher
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.cnn import Conv3d, xavier_init
from models.utils.loss_utils import calc_semantic_loss
import copy
from models.layers.spatial_cross_attention import MSDeformableAttention3D
from mmdet.models.utils.transformer import inverse_sigmoid
from .det_head import VoxelAttentionAggregation

@HEADS.register_module()
class FoSOccHead(nn.Module): 
    def __init__(self,
                 in_channels=32,
                 embed_dims=256, # embed_dims = 256, need to be fixed
                 num_stuff_queries=50,
                 out_channels=256, # out_channels = 256, need to be fixed
                 num_classes=16,
                 inst_class_ids=[1, 2, 3, 4],
                 num_decoder_layers=1,
                 with_reg_branches=True,
                 with_velo=False,
                 class_weight_multiclass=None,
                 semantic_loss_flag=True,
                 sem_loss_weight=1,
                 transformer=None,
                 transformer_decoder=None,
                 positional_encoding=None,
                 occ_size=[200, 200, 16],
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 loss_cfgs=None,
                 **kwargs):
        super(FoSOccHead, self).__init__()
        self.fp16_enabled = False

        self.num_stuff_queries = num_stuff_queries
        self.num_classes = num_classes
        self.inst_class_ids = inst_class_ids
        self.pc_range = pc_range
        self.occ_size = occ_size
        self.with_velo = with_velo

        self.class_weight_multiclass = class_weight_multiclass

        self.criterions = {k: build_loss(loss_cfg) for k, loss_cfg in loss_cfgs.items()}
        
        self.matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)

        self.sem_loss_weight = sem_loss_weight
        self.semantic_loss_flag = semantic_loss_flag
        if semantic_loss_flag:
            self.seg_pred_heads = nn.ModuleList()
            self.seg_pred_heads.append(nn.Sequential(
                                nn.Linear(embed_dims, in_channels*2),
                                nn.Softplus(),
                                nn.Linear(in_channels*2, num_classes),
                            ))

        self.cls_embed = nn.Linear(embed_dims, self.num_classes)
        self.mask_embed = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, out_channels))
        # the content below from cotr
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.transformer = build_transformer(transformer) if transformer is not None else None
        self.positional_encoding = build_positional_encoding(positional_encoding) if positional_encoding is not None else None
        self.decoder_input_projs = Conv3d(in_channels, embed_dims, kernel_size=1)
        self.enhance_projs = Conv3d(embed_dims, in_channels, kernel_size=1)
        self.offsets_conv = Conv3d(embed_dims, 7, kernel_size = 3, padding = 1) # here I may change it to 6, which must be 7
        #self.offsets_7to1 = Conv3d(7, 1, kernel_size=3, padding=1)
        enhance_channel = in_channels * 2
        
        self.enhance_conv = ConvModule(
                        enhance_channel,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        #self.q_proj = nn.Linear(6, embed_dims)
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)#MaskOccDecoder
        #self.dca = MSDeformableAttention3D(embed_dims = 256) # may need fixed
        #self.offsets_aggregator = nn.Conv3d(in_channels=6 + embed_dims, out_channels=embed_dims, kernel_size=3, padding = 1)
        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(in_channels*5,in_channels*2,(3,3,1),padding=(1,1,0)),
            nn.BatchNorm3d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*2, in_channels*2, (2, 2, 1), stride=(2, 2, 1)),
            nn.BatchNorm3d(in_channels*2),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels*4,in_channels*2,(3,3,1),padding=(1,1,0)),
            nn.BatchNorm3d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*2, in_channels*2, (2, 2, 1), stride=(2, 2, 1)),
            nn.BatchNorm3d(in_channels*2),
            nn.ReLU(inplace=True),
        )
        self.n = 200 # need to be named
        #self.AttnNorm = nn.LayerNorm(self.n)
        #self.query_norm = nn.LayerNorm(self.n)
        #self.key_norm = nn.LayerNorm(self.n)
        #self.value_norm = nn.LayerNorm(self.n)
        self.down_sampling_ratio = 0.5 # may need fixed
        self.refine_conv = nn.ModuleList()
        self.num_inst_layer = 1 
        dropout_rate=0.0
        use_bias = False
        align_corners=False
        kernel_size=1
        num_groups=32
        
        for i in range(self.num_inst_layer):
            
            self.refine_conv.append( 
                VoxelAttentionAggregation(
                    #in_channels=[128,512,1024,2048][0], # need to be fixed
                    in_channels=32, # need to be fixed
                    num_directions=7, # here must be 7
                    dropout_rate=dropout_rate,
                    use_bias=use_bias,
                    align_corners=align_corners,
                    kernel_size=kernel_size,
                    num_groups=num_groups)
                )
        
        #self.final_conv = nn.Conv3d(kernel_size=3, padding = 1, stride = 1, in_channels = in_channels * 3, out_channels = 128)
        
        self.back_conv = ConvModule(
                        96,
                        embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        
        self.final_conv = ConvModule(
                        in_channels*3,
                        embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        
        self.final_conv1 = ConvModule(
                        embed_dims,
                        64,
                        kernel_size=1,
                        stride=1,
                        #padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        
        self.stuff_embedding = nn.Embedding(self.num_stuff_queries, self.embed_dims*2)

        self.ref3d_predictor = nn.Linear(self.embed_dims, 3)

        self.compact_proj = Conv3d(in_channels, embed_dims, kernel_size=1)
        d_model = 3 # need to be fixed
        #self.query_proj = nn.Linear(d_model, d_model)
        #self.key_proj   = nn.Linear(d_model, d_model)
        #self.value_proj = nn.Linear(d_model, d_model) #Here I may comment the value_proj
        self.reg_branches = None
        if with_reg_branches:
            reg_branch = nn.Linear(embed_dims, 3)
            self.reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)]) if num_decoder_layers else None
            
        if self.with_velo:
            assert with_reg_branches == True, 'with_reg_branches should be True when with_velo is True'
            self.velo_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)]) if num_decoder_layers else None
        
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.transformer is not None:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        xavier_init(self.ref3d_predictor, distribution='uniform', bias=0.)

    def seg_voxels(self, voxel_decoder_outputs):
        seg_pred_list = []
        i, voxel_decoder_output = -1, voxel_decoder_outputs[-1]
        seg_pred = voxel_decoder_output.permute(0, 2, 3, 4, 1) # to [B, W, H, D, CLS]
        seg_pred = self.seg_pred_heads[i](seg_pred)
        seg_pred = seg_pred.permute(0, 4, 1, 2, 3)# to [B, CLS, W, H, D]
        seg_pred_list.append(seg_pred)
        return seg_pred_list

    def enhance_occ_feature(self, img_feats, compact_occ, ms_occ_feature, img_metas, cam_params, **kwargs):
        def create_grid(grid_x, grid_y, grid_z, X, Y, Z, device):
            grid_x_norm = 2.0 * grid_x / (X - 1) - 1.0
            grid_y_norm = 2.0 * grid_y / (Y - 1) - 1.0
            grid_z_norm = 2.0 * grid_z / (Z - 1) - 1.0
            grid = torch.stack((grid_x_norm, grid_y_norm, grid_z_norm), dim=-1)
            return grid
        def generate_grid(offsets):
            B, C, X, Y, Z = offsets.shape
            device = offsets.device

            # Create voxel grid coordinates with shape [B, X, Y, Z]
            grid_x = torch.arange(X, device=device).view(1, X, 1, 1).expand(B, X, Y, Z).float()
            grid_y = torch.arange(Y, device=device).view(1, 1, Y, 1).expand(B, X, Y, Z).float()
            grid_z = torch.arange(Z, device=device).view(1, 1, 1, Z).expand(B, X, Y, Z).float()

            # Recover offsets from normalization.
            # Since offsets are normalized by dividing by X, Y, Z respectively,
            # multiply back by (X - 1), (Y - 1) and (Z - 1) to recover the real offset magnitude.
            offset_x_pos = offsets[:, 0] * (X - 1)
            offset_x_neg = offsets[:, 1] * (X - 1)
            offset_y_pos = offsets[:, 2] * (Y - 1)
            offset_y_neg = offsets[:, 3] * (Y - 1)
            offset_z_pos = offsets[:, 4] * (Z - 1)
            offset_z_neg = offsets[:, 5] * (Z - 1)

            # Create sampling grids for six directions by updating only one coordinate component
            coords_list = []

            sample_x = torch.clamp(grid_x + offset_x_pos.float(), 0, X - 1)
            coords_list.append(create_grid(sample_x, grid_y, grid_z, X, Y, Z, device))

            sample_x = torch.clamp(grid_x - offset_x_neg.float(), 0, X - 1)
            coords_list.append(create_grid(sample_x, grid_y, grid_z, X, Y, Z, device))

            sample_y = torch.clamp(grid_y + offset_y_pos.float(), 0, Y - 1)
            coords_list.append(create_grid(grid_x, sample_y, grid_z, X, Y, Z, device))

            sample_y = torch.clamp(grid_y - offset_y_neg.float(), 0, Y - 1)
            coords_list.append(create_grid(grid_x, sample_y, grid_z, X, Y, Z, device))

            sample_z = torch.clamp(grid_z + offset_z_pos.float(), 0, Z - 1)
            coords_list.append(create_grid(grid_x, grid_y, sample_z, X, Y, Z, device))

            sample_z = torch.clamp(grid_z - offset_z_neg.float(), 0, Z - 1)
            coords_list.append(create_grid(grid_x, grid_y, sample_z, X, Y, Z, device))

            coords_list.append(offsets[:,6].unsqueeze(-1)) #here, disabled when offsets_channel = 6
            return coords_list
        bs = compact_occ.shape[0]
        dtype = compact_occ.dtype
        occ_x, occ_y, occ_z = compact_occ.shape[-3:] # [50, 50, 16]

        occ_queries = compact_occ
        # project bev feature to the transformer embed dims
        if self.in_channels != self.embed_dims:
            occ_queries = self.decoder_input_projs(occ_queries)
        
        # begin-----------------Encoder---------------------
        occ_queries = occ_queries.flatten(2).permute(0, 2, 1) # [b, x*y*z, C]
        occ_mask = torch.zeros((bs, occ_x, occ_y, occ_z),
                            device=occ_queries.device).to(dtype)
        occ_pos = self.positional_encoding(occ_mask, 1).to(dtype)

        enhanced_occ_feature = self.transformer(
                                    img_feats,
                                    occ_queries,
                                    occ_x,
                                    occ_y,
                                    occ_z,
                                    occ_pos=occ_pos,
                                    img_metas=img_metas,
                                    cam_params=cam_params[:],
                                    **kwargs) # [b, h*w, c]
        enhanced_occ_feature = enhanced_occ_feature.permute(0, 2, 1).contiguous().view(bs, -1, occ_x, occ_y, occ_z)
        enhanced_occ_feature = self.enhance_projs(enhanced_occ_feature) # [b, 32, x, y, z]
        occ_cat = torch.cat((compact_occ, enhanced_occ_feature), axis=1)
        compact_occ = self.enhance_conv(occ_cat)
        

        # end-----------------Encoder---------------------
        occ2, occ1, occ0 = ms_occ_feature
        occ_0 = self.up0(torch.cat([compact_occ, occ0], dim=1)) # (50, 50, 16) to (64, 100, 100, 16)
        occ1 = F.interpolate(occ1, size=occ_0.shape[-3:],
                            mode='trilinear', align_corners=True)# (100, 100, 8) to (100, 100, 16)
        
        occ_1 = self.up1(torch.cat([occ_0, occ1], dim=1)) # (100, 100, 16) to (64, 200, 200, 16)
        fullres_occ = self.final_conv(torch.cat([occ2, occ_1], dim=1))#.permute(0, 1, 4, 3, 2) # bczhw -> bcwhz
        
        # ---This part is about how predicted offsets & focus feature help to focus geometry details
        offsets = self.offsets_conv(fullres_occ) #[B, 6, W, H, D]
        fullres_occ = self.final_conv1(fullres_occ)
        
        offsets = offsets.sigmoid()
        offsets = F.interpolate(offsets, scale_factor=(0.5, 0.5, 1), mode='nearest')

        grid_list = generate_grid(offsets)
        for i in range(self.num_inst_layer):
                compact_occ = self.refine_conv[i](compact_occ, grid_list, self.down_sampling_ratio)
        
        fullres_occ = self.back_conv(torch.cat([fullres_occ, F.interpolate(compact_occ, scale_factor=(4, 4, 1), mode='nearest')], dim = 1))
        #---
        return compact_occ, fullres_occ, offsets
        #return compact_occ, fullres_occ

    @auto_fp16(apply_to=('thing_query_embeds', 'img_feats', 'compact_occ', 'ms_occ_feature'), out_fp32=True)
    def forward(self, thing_query_embeds, thing_ref_pts, img_feats, compact_occ, ms_occ_feature, cam_params, info_mask=None, img_metas=None,**kwargs):
            #ms_occ_feature = volume_feats
        
        compact_occ, fullres_occ, offsets = self.enhance_occ_feature(img_feats, compact_occ, 
                                                            ms_occ_feature, img_metas, cam_params, **kwargs)
 
        bs = compact_occ.shape[0] #[bs, 50, 50, 16]
        dtype = compact_occ.dtype
        
        # -----------------Decoder---------------------

        stuff_query_embeds = self.stuff_embedding.weight.to(dtype)

        stuff_query_pos, stuff_query = torch.split(
            stuff_query_embeds, self.embed_dims, dim=1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(bs, -1, -1)
        stuff_query = stuff_query.unsqueeze(0).expand(bs, -1, -1)
        stuff_ref_pts = self.ref3d_predictor(stuff_query_pos)[:, :, None, :]# [bs, num_queries, num_level, 3]
        stuff_ref_pts = stuff_ref_pts.sigmoid() # 0-1
        
        occ_value = self.compact_proj(compact_occ)
        
        x_len, y_len, z_len = occ_value.shape[-3:]
        occ_value = occ_value.flatten(-3).permute(2, 0, 1) # [xyz, bs, c]
        decoder_spatial_shapes = torch.tensor([
                [x_len, y_len, z_len],
            ], device=stuff_query.device)    
        lsi = torch.tensor([0,], device=stuff_query.device)

        # [bs, n, c] --> [n, bs, c]
        stuff_query = stuff_query.permute(1, 0, 2)
        stuff_query_pos = stuff_query_pos.permute(1, 0, 2)

        thing_query_embeds = thing_query_embeds.unsqueeze(1).expand(-1, bs, -1)
        thing_query_pos, thing_query = torch.split(
            thing_query_embeds, self.embed_dims, dim=-1)
        
        thing_ref_pts = thing_ref_pts.unsqueeze(0).expand(bs, -1, -1).unsqueeze(-2)
        thing_ref_pts = thing_ref_pts.sigmoid() # 0-1
        
        query_embeds = torch.cat([thing_query, stuff_query], dim=0)
        query_pos = torch.cat([thing_query_pos, stuff_query_pos], dim=0)
        query_ref_pts = torch.cat([thing_ref_pts, stuff_ref_pts], dim=1)
        
        decoder_outs, inter_refer_pts = self.transformer_decoder( # may could be improved
                            query=query_embeds,
                            key=None,
                            value=occ_value,
                            query_pos=query_pos,
                            reference_points=query_ref_pts, 
                            spatial_shapes=decoder_spatial_shapes,
                            level_start_index=lsi,
                            reg_branches=self.reg_branches,
                            **kwargs
                        )
        last_ref_points = inter_refer_pts[-1].squeeze(2)[:, :-self.num_stuff_queries].clone()
        last_ref_points = inverse_sigmoid(last_ref_points) # [bs, num_queries, 3]
        last_thing_embeds = decoder_outs[-1].permute(1, 0, 2)[:, :-self.num_stuff_queries] # [bs, num_queries, c]
        
        cls_pred_list = []
        mask_pred_list = []
        stuff_cls_pred_list = []
        stuff_mask_pred_list = []
        ref_pts_and_velo_list = []
        #decoder: [n, bs, embed_dims]
        for lvl, decoder_out in enumerate(decoder_outs):
            cls_pred, mask_pred = self._forward_head(decoder_out.transpose(0, 1), fullres_occ)
            cls_pred_list.append(cls_pred[:, :-self.num_stuff_queries])
            mask_pred_list.append(mask_pred[:, :-self.num_stuff_queries])
            stuff_cls_pred_list.append(cls_pred[:, -self.num_stuff_queries:])
            stuff_mask_pred_list.append(mask_pred[:, -self.num_stuff_queries:])

            out_ref_pts = inter_refer_pts[lvl].squeeze(2)[:, :-self.num_stuff_queries].clone() # in 0-1
            # use self.pc_range to convert to real world
            out_ref_pts[..., 0:1] = (out_ref_pts[..., 0:1] *
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            out_ref_pts[..., 1:2] = (out_ref_pts[..., 1:2] *
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            out_ref_pts[..., 2:3] = (out_ref_pts[..., 2:3] *
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]) 
            if self.with_velo:
                ref_velo_pred = self.velo_branches[lvl](decoder_out[:-self.num_stuff_queries]).permute(1, 0, 2)
                out_ref_pts = torch.cat([out_ref_pts, ref_velo_pred], dim=-1)
            
            ref_pts_and_velo_list.append(out_ref_pts) # 3 or 6

        seg_pred_list = []
        if self.semantic_loss_flag:
            seg_pred_list = self.seg_voxels([fullres_occ])
        offsets_pred_list = [offsets]
        not_track_dict = {
            'seg_pred_list': seg_pred_list,
            'stuff_cls_pred_list': stuff_cls_pred_list,
            'stuff_mask_pred_list': stuff_mask_pred_list,
            'offsets_pred_list': offsets_pred_list,
        }
        return not_track_dict, cls_pred_list, mask_pred_list, ref_pts_and_velo_list, last_thing_embeds, last_ref_points

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, x, y, z).
            attn_mask_target_size (tuple[int, int,]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """

        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)

        return cls_pred, mask_pred
    

    @force_fp32(apply_to=('seg_pred_list', 'stuff_cls_pred_list', 'stuff_mask_pred_list'))
    def loss(self, 
             seg_pred_list, 
             stuff_cls_pred_list, 
             stuff_mask_pred_list, 
             voxel_semantics, 
             offsets_pred_list,
             gt_offsets,  
             visible_mask=None):
        loss_dict = {}
        B = voxel_semantics.shape[0]

        if visible_mask is not None:
            assert visible_mask.shape == voxel_semantics.shape
            assert visible_mask.dtype == torch.bool
        
        if self.semantic_loss_flag and len(seg_pred_list):
            for i, seg_pred in enumerate(seg_pred_list):
                loss_dict_i = {}
                for b in range(B):
                    loss_dict_i_b = calc_semantic_loss(
                        voxel_semantics[b:b + 1],
                        seg_pred[b:b + 1],
                        visible_mask[b:b + 1] if visible_mask is not None else None,
                        class_weights=self.class_weight_multiclass,
                        sem_loss_weight=self.sem_loss_weight,
                    )
                    for loss_key in loss_dict_i_b.keys():
                        loss_dict_i[loss_key] = loss_dict_i.get(loss_key, 0) + loss_dict_i_b[loss_key] / B

                for k, v in loss_dict_i.items():
                    loss_dict['%s_%d' % (k, i)] = v

        voxel_semantics = voxel_semantics.reshape(B, -1)  # [B, N]

        if visible_mask is not None:
            visible_mask = visible_mask.reshape(B, -1)  # [B, N]
        
        stuff_instances = []
        stuff_class_ids = []
        for b_idx in range(B):
            stuff_count = 0
            stuff_class_ids_b = []
            stuff_instances_b = torch.ones_like(voxel_semantics[0]) * 255
            interest_mask_b = visible_mask[b_idx]
            mask_stuff_instances = stuff_instances_b[interest_mask_b]
            mask_semantics = voxel_semantics[b_idx][interest_mask_b]
            
            for class_id in range(self.num_classes - 1): # NOTE: we treat free label as negtive label. 
                if torch.sum(mask_semantics == class_id) == 0:
                    continue

                if class_id not in self.inst_class_ids:
                    mask_stuff_instances[mask_semantics == class_id] = stuff_count
                    stuff_count += 1
                    stuff_class_ids_b.append(class_id)
            stuff_instances_b[interest_mask_b] = mask_stuff_instances

            stuff_instances.append(stuff_instances_b)
            stuff_class_ids.append(torch.as_tensor(stuff_class_ids_b, dtype=torch.int64).to(voxel_semantics.device))

        stuff_instances = torch.stack(stuff_instances)
        
        for i, pred in enumerate(stuff_mask_pred_list):
            pred[torch.isnan(pred)] = 0
            pred[torch.isinf(pred)] = 0
            assert torch.isnan(pred).sum().item() == 0
            assert torch.isinf(pred).sum().item() == 0
            indices = self.matcher(pred, stuff_cls_pred_list[i], stuff_instances, stuff_class_ids, visible_mask)
            
            loss_mask, loss_dice, loss_class, loss_offsets = self.criterions['loss_mask2former'](
                pred, stuff_cls_pred_list[i], stuff_instances, stuff_class_ids, offsets_pred_list[0], gt_offsets, indices, visible_mask)
            loss_dict['loss_stuff_mask_{:d}'.format(i)] = loss_mask
            loss_dict['loss_stuff_dice_mask_{:d}'.format(i)] = loss_dice
            loss_dict['loss_stuff_class_{:d}'.format(i)] = loss_class
            loss_dict['loss_offsets_{:d}'.format(i)] = loss_offsets # weighted L1 loss

        return loss_dict