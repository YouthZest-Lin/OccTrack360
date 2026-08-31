import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES, build_loss
from mmdet.core import reduce_mean

def calc_semantic_loss(voxel_semantics, seg_pred_dense, visible_mask=None, class_weights=None, sem_loss_weight=1.0):
    loss_dict = dict()
    seg_pred_dense[torch.isnan(seg_pred_dense)] = 0
    seg_pred_dense[torch.isinf(seg_pred_dense)] = 0
    
    voxel_semantics = voxel_semantics.long() 
    # May need fixed
    
    if seg_pred_dense is not None:  # semantic prediction
        num_classes = seg_pred_dense.shape[1] # [B, CLS, W, H, D]
        if seg_pred_dense.shape[-1] != voxel_semantics.shape[-1] or seg_pred_dense.shape[-2] != voxel_semantics.shape[-2]\
            or seg_pred_dense.shape[-3] != voxel_semantics.shape[-3]:
            seg_pred_dense = F.interpolate(seg_pred_dense, size=voxel_semantics.shape[-3:], mode='trilinear', align_corners=False)
        seg_pred_dense = seg_pred_dense.permute(0, 2, 3, 4, 1)   # [B, X, Y, Z, CLS]
        if visible_mask is not None:
            seg_pred_dense = seg_pred_dense[visible_mask]
            voxel_semantics = voxel_semantics[visible_mask]

        # not mask out the free space
        seg_pred_dense = seg_pred_dense.view(-1, num_classes)
        voxel_semantics = voxel_semantics.view(-1)

        class_weights=torch.tensor(class_weights).to(seg_pred_dense.device)
        loss_dict['loss_sem'] = F.cross_entropy(seg_pred_dense, voxel_semantics, weight=class_weights) * sem_loss_weight

    return loss_dict

# borrowed from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L21
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    mask_camera: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if mask_camera is not None:
        inputs = inputs[:, :, mask_camera]
        targets = targets[:, :, mask_camera]

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.squeeze(1)
    
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


# dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


# borrowed from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L48
def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    mask_camera: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # [M, 1, K]
    if mask_camera is not None:
        mask_camera = mask_camera.to(torch.int32)
        mask_camera = mask_camera[None, None, ...].expand(
            targets.shape[0], 1, mask_camera.shape[-1]
        )
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, mask_camera, reduction="none"
        )
    else:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(2).mean(1).sum() / num_masks


# sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


# modified from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L90
@LOSSES.register_module()
class Mask2Former3DLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        loss_cls_weight=1.0,
        loss_mask_weight=1.0,
        loss_dice_weight=1.0,
        class_balanced_weight=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        if class_balanced_weight is None:
            self.class_balanced_weight = torch.ones(self.num_classes)
        else:
            assert len(class_balanced_weight) == num_classes, "class_balanced_weight should have the same length as num_classes"
            self.class_balanced_weight = class_balanced_weight
        self.loss_cls = build_loss(
            dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            )
        )
        self.loss_reg = build_loss(
            dict(
                type="SmoothL1Loss",     # 或 L1Loss、MSELoss、BalancedL1Loss 等
                beta=1.0,                # SmoothL1 的平滑参数（可选）
                loss_weight=1,         # 损失权重
                reduction='mean'         # 可选：'mean'、'sum'、'none'
            )
        )
    


    def forward(self, mask_pred, class_pred, mask_gt, class_gt, offsets_pred, gt_offsets, indices, mask_camera):
        bs = mask_pred.shape[0]
        loss_masks = torch.tensor(0).to(mask_pred)
        loss_dices = torch.tensor(0).to(mask_pred)
        loss_classes = torch.tensor(0).to(mask_pred)

        num_total_pos = sum([tc.numel() for tc in class_gt])
        avg_factor = torch.clamp(
            reduce_mean(class_pred.new_tensor([num_total_pos * 1.0])), min=1
        ).item()  # TODO

        for b in range(bs):
            mask_camera_b = mask_camera[b] if mask_camera is not None else None  # N
            tgt_mask = mask_gt[b]
            num_instances = class_gt[b].shape[0]

            tgt_class = class_gt[b]
            tgt_mask = tgt_mask.unsqueeze(-1) == torch.arange(num_instances).to(
                mask_gt.device
            )
            tgt_mask = tgt_mask.permute(1, 0)

            src_idx, tgt_idx = indices[b]
            src_mask = mask_pred[b][
                src_idx
            ]  # [M, N], M is number of gt instances, N is number of remaining voxels
            tgt_mask = tgt_mask[tgt_idx]  # [M, N]
            src_class = class_pred[b]  # [Q, CLS]

            # pad non-aligned queries' tgt classes with 'no object'(also named as free)
            pad_tgt_class = torch.full(  #
                (src_class.shape[0],),
                self.num_classes - 1,
                dtype=torch.int64,
                device=class_pred.device,
            )  # [Q]
            pad_tgt_class[src_idx] = tgt_class[tgt_idx]

            # only calculates loss mask for aligned pairs
            loss_mask, loss_dice = self.loss_masks(
                src_mask, tgt_mask, avg_factor=avg_factor, mask_camera=mask_camera_b
            )
            # calculates loss class for all queries
            loss_class = self.loss_labels(
                src_class,
                pad_tgt_class,
                self.class_balanced_weight.to(src_class.device),
                avg_factor=avg_factor,
            )
            offsets_pred = offsets_pred.view(7, -1) # must be (7, -1)
            gt_offsets = F.interpolate(gt_offsets, scale_factor=(0.5,0.5,1),mode = 'nearest')
            '''
            for i in range(offsets_pred.shape[1]):
                offsets_pred[:,i] = offsets_pred[:,i] / offsets_pred[:,i].max()
            
            for i in range(gt_offsets.shape[1]):
                gt_offsets[:,i] = gt_offsets[:,i] / gt_offsets[:,i].max()
            '''
            gt_offsets = gt_offsets.view(7, -1)#here must be (7, -1)
            def cal_loss_offsets(pred, gt, mask, k1 = 10, k2 = 1000):
                if mask.shape[0] == 128*128*32:
                    mask = mask.reshape([1, 1, 128, 128, 32]).to(torch.float16) # here should depend on dataset
                if mask.shape[0] == 200*200*16:
                    mask = mask.reshape([1, 1, 200, 200, 16]).to(torch.float16) # here should depend on dataset
                mask = F.interpolate(mask, scale_factor=(0.5,0.5,1),mode = 'nearest')
                mask = mask.view(-1)
                weight = (gt == 0) * k1 + (gt != 0) * k2
                return self.loss_reg(pred * weight * mask, gt * weight * mask)
            #print(f'pred_max:{offsets_pred.max()}')
            #print(f'gt_max:{gt_offsets.max()}')
            #print(f'pred_min:{offsets_pred.min()}')
            #print(f'gt_min:{gt_offsets.min()}')
            loss_offsets = cal_loss_offsets(offsets_pred, gt_offsets, mask_camera_b) #here I may mask the loss_offsets
            #loss_offsets = 0
            loss_masks += loss_mask * self.loss_mask_weight
            loss_dices += loss_dice * self.loss_dice_weight
            loss_classes += loss_class * self.loss_cls_weight

        return loss_masks, loss_dices, loss_classes, loss_offsets

    # mask2former use point sampling to calculate loss of fewer important points
    # we omit point sampling as we have limited number of points
    def loss_masks(self, src_mask, tgt_mask, avg_factor=None, mask_camera=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        num_masks = tgt_mask.shape[0]
        src_mask = src_mask.view(num_masks, 1, -1)
        tgt_mask = tgt_mask.view(num_masks, 1, -1)

        if avg_factor is None:
            avg_factor = num_masks

        loss_dice = dice_loss(src_mask, tgt_mask, avg_factor, mask_camera)
        loss_mask = sigmoid_ce_loss(src_mask, tgt_mask.float(), avg_factor, mask_camera)

        return loss_mask, loss_dice

    def loss_labels(self, src_class, tgt_class, class_balanced_weight=None, avg_factor=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        loss = self.loss_cls(
            src_class, 
            tgt_class, 
            torch.ones_like(tgt_class), # TODO: class_balanced_weight.to(tgt_class.device),
            avg_factor=avg_factor
        )
        return loss#.mean()
    


# borrowed from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L21
def custom_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    mask_camera: torch.Tensor,
    distance_weight: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if mask_camera is not None:
        inputs = inputs[:, :, mask_camera]
        targets = targets[:, :, mask_camera]

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.squeeze(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


# dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


# borrowed from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L48
def custom_sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    mask_camera: torch.Tensor,
    distance_weight: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # [M, 1, K]
    if mask_camera is not None:
        # mask_camera = mask_camera.to(torch.int32)
        if distance_weight is not None:
            mask_camera = mask_camera * distance_weight
        mask_camera = mask_camera[None, None, ...].expand(
            targets.shape[0], 1, mask_camera.shape[-1]
        )
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, mask_camera, reduction="none"
        )
    else:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(2).mean(1).sum() / num_masks