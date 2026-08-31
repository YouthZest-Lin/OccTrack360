from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, pc_range=None, grid_config=None, data_config=None,
                 fix_bug=False, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(OccEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        
        if grid_config is not None:
            self.x_bound = grid_config['x']
            self.y_bound = grid_config['y']
            self.z_bound = grid_config['z']
        if data_config is not None:
            self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.fix_bug = fix_bug

    @staticmethod
    def get_reference_points(X, Y, Z, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        #occ_x, occ_y, occ_z = compact_occ.shape[-3:] # [50, 50, 16]
        xs = torch.linspace(0.5, X - 0.5, X, dtype=dtype,
                            device=device).view(X, 1, 1).expand(X, Y, Z) / X
        ys = torch.linspace(0.5, Y - 0.5, Y, dtype=dtype,
                            device=device).view(1, Y, 1).expand(X, Y, Z) / Y
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(1, 1, Z).expand(X, Y, Z) / Z
        ref_3d = torch.stack((xs, ys, zs), -1) # (X, Y, Z, 3)
        ref_3d = ref_3d.flatten(0, 2) # (X*Y*Z, 3)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)# (bs, num_level, X*Y*Z, 3)
        return ref_3d

    def custom_get_reference_points(self, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1]/2
        Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1]/2
        Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1]/2
        X, Y, Z = torch.meshgrid([X, Y, Z])
        coords = torch.stack([X, Y, Z], dim=-1)
        coords = coords.to(dtype).to(device)
        # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)
        return coords

    @force_fp32(apply_to=('reference_points', 'cam_params', 'img_metas'))
    def cumstom_point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        rots, trans, intrins, cal, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.final_dim
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points -= trans.view(B, N, 1, 1, 1, 3)

        xi = cal[:,:,0]
        k1 = cal[:,:,1]
        k2 = cal[:,:,2]
        p1 = cal[:,:,3]
        p2 = cal[:,:,4]

        
        
        #combine = rots.matmul(torch.inverse(intrins)).inverse()

        
        #reference_points_cam = torch.inverse(rots).view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        #reference_points_cam = intrins.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        #reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        
        reference_points_cam = torch.inverse(rots).view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        
        reference_points_cam[...,0] = reference_points_cam[...,0] / (reference_points_cam[...,2] + 1e-8)
        reference_points_cam[...,1] = reference_points_cam[...,1] / (reference_points_cam[...,2] + 1e-8)
        lamb = 1/(reference_points_cam[...,2] + xi.view(B, N, 1, 1, 1) * torch.sqrt(reference_points_cam[...,0]**2 + reference_points_cam[...,1]**2 + reference_points_cam[...,2]**2))
        reference_points_cam[...,0] = lamb * reference_points_cam[...,0]
        reference_points_cam[...,1] = lamb * reference_points_cam[...,1]
        
        x = reference_points_cam[...,0]
        y = reference_points_cam[...,1]
        r = torch.sqrt(x**2 + y**2)

        reference_points_cam[...,0] = x * (1 + k1.view(B, N, 1, 1, 1) * r**2 + k2.view(B, N, 1, 1, 1) * r**4) + 2 * p1.view(B, N, 1, 1, 1) * x * y + p2.view(B, N, 1, 1, 1) * (r**2 + 2*x**2)
        reference_points_cam[...,1] = y * (1 + k1.view(B, N, 1, 1, 1) * r**2 + k2.view(B, N, 1, 1, 1) * r**4) + 2 * p2.view(B, N, 1, 1, 1) * x * y + p1.view(B, N, 1, 1, 1) * (r**2 + 2*y**2)
        
        reference_points_cam = intrins.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        
        #reference_points_cam = rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        
        #print(f'reference_points_cam[:,:,:,:,:,0].max():{reference_points_cam[:,:,:,:,:,0].max()}')
        #print(f'reference_points_cam[:,:,:,:,:,1].max():{reference_points_cam[:,:,:,:,:,1].max()}')
        #print(f'reference_points_cam[:,:,:,:,:,2].max():{reference_points_cam[:,:,:,:,:,2].max()}')
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3) 
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps) 
                 & (reference_points_cam[..., 0:1] < (1.0-eps)) 
                 & (reference_points_cam[..., 1:2] > eps) 
                 & (reference_points_cam[..., 1:2] < (1.0-eps)))
        B, N, X, Y, Z, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, X*Y*Z, 3)
        reference_points_cam = reference_points_cam[:, :, :, None, :]# (N, B, X*Y*Z, num_level, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, X*Y*Z, 1)

        return reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]


    @auto_fp16()
    def forward(self,
                occ_query,
                key,
                value,
                *args,
                occ_x=None,
                occ_y=None,
                occ_z=None,
                occ_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                cam_params=None,
                valid_ratios=None,
                prev_occ=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = occ_query
        intermediate = []
        
        absolute_ref_3d = self.custom_get_reference_points(occ_query.size(1), 
                device=occ_query.device, dtype=occ_query.dtype)# [X, Y, Z, 3]
        reference_points_cam, occ_mask, occ_query_depth = self.cumstom_point_sampling(
                absolute_ref_3d, self.pc_range, kwargs['img_metas'], cam_params=cam_params)
        # (bs, num_level, X*Y*Z, 3) for self-attention
        ref_3d = self.get_reference_points(occ_x, occ_y, occ_z, bs=occ_query.size(1), 
                device=occ_query.device, dtype=occ_query.dtype) 
        # (num_query, bs, embed_dims) -> (bs, num_query(hwz), embed_dims)
        occ_query = occ_query.permute(1, 0, 2) 
        occ_pos = occ_pos.permute(1, 0, 2) 

        for lid, layer in enumerate(self.layers):
            output = layer(
                occ_query,
                key,
                value,
                *args,
                occ_pos=occ_pos,
                ref_3d=ref_3d, # [bs, num_level, X*Y*Z, 3]
                occ_x=occ_x,
                occ_y=occ_y,
                occ_z=occ_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam, # [num_cam, bs, XYZ, num_level, 2]
                occ_mask=occ_mask,                         # [num_cam, bs, XYZ, 1]num_level
                prev_occ=prev_occ,
                **kwargs)

            occ_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class OccFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default: 2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(OccFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                occ_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                occ_x=None,
                occ_y=None,
                occ_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_occ=None,
                occ_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # deformable self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity=query,
                    query_pos=occ_pos,
                    key_pos=occ_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_3d.permute(0, 2, 1, 3), # -> [bs, xyz, num_level, 3]
                    spatial_shapes=torch.tensor(
                        [[occ_x, occ_y, occ_z]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    occ_mask=occ_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
