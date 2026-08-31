import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from typing import List
from models.utils.structures import Instances
from mmdet.core import build_assigner
from mmdet.models import build_loss
from mmdet.models.builder import LOSSES
from mmdet.core import reduce_mean
from mmcv.runner import force_fp32
from models.utils.matcher import HungarianMatcherMix
from .loss_utils import sigmoid_ce_loss, dice_loss


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@LOSSES.register_module()
class ClipMatcher(nn.Module):
    # modified from https://github.com/megvii-model/MOTR/blob/main/models/motr.py#L38
    def __init__(self, 
                 num_classes,
                 center_supervise=False,
                 code_weights=[1.0, 1.0, 1.0, 0.2, 0.2, 0.2], 
                 loss_cls_weight=1.0,
                 loss_center_weight=1.0,
                 loss_mask_weight=1.0,
                 loss_dice_weight=1.0,
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.center_supervise = center_supervise
        self.register_buffer('code_weights', torch.tensor(code_weights,
                                                          requires_grad=False))
        self.loss_cls_weight = loss_cls_weight
        self.loss_center_weight = loss_center_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        self.matcher = HungarianMatcherMix(stuff_flag=False, cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
        self.loss_cls = build_loss(
            dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            )
        )
        self.loss_l1 = build_loss(dict(type='L1Loss', loss_weight=0.25))
        if self.center_supervise:
            self.losses = ['labels', 'masks', 'centers']
        else:
            self.losses = ['labels', 'masks']
        self.losses_dict = {}
        self._current_frame_idx = 0
        # self.fp16_enabled = False
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        #print(self._current_frame_idx)
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_masks=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_masks(self, num_samples):
        num_masks = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        return num_masks

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty masks
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v.labels) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_loss(self, loss, outputs, gt_instances, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'masks': self.loss_masks,
            'centers': self.loss_centers,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, **kwargs)

    def loss_centers(self, outputs, gt_instances: List[Instances], indices: List[tuple]):
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_centers = outputs['pred_centers'][idx]
        target_centers = torch.cat([gt_per_img.centers[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        #from IPython import embed
        #embed()

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        # [num_matched]
        mask = (target_obj_ids != -1)
        code_len = target_centers.shape[-1]
        center_weights = torch.ones_like(target_centers) * self.code_weights[:code_len]
        # not all the velo of centers are valid
        invalid_mask = torch.isnan(target_centers)
        center_weights[invalid_mask] = 0
        target_centers[invalid_mask] = 0
        avg_factor = src_centers[mask].size(0)
        avg_factor = reduce_mean(
                target_centers.new_tensor([avg_factor]))
        
        loss_center = self.loss_l1(src_centers[mask], target_centers[mask],
                                     center_weights[mask],
                                     avg_factor=avg_factor.item())

        losses = {}
        losses['loss_center'] = loss_center * self.loss_center_weight

        return losses

    def loss_masks(self, outputs, gt_instances: List[Instances], indices: List[tuple]):
        """Compute the losses related to the bounding masks, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_masks, 4]
           The target masks are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        losses = {}
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_masks = outputs['pred_masks'][idx]
        target_masks = torch.cat([gt_per_img.masks[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        #from IPython import embed
        #embed()

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        # [num_matched]
        mask = (target_obj_ids != -1)

        avg_factor = src_masks[mask].size(0)
        avg_factor = reduce_mean(
                src_masks.new_tensor([avg_factor]))
        mask_camera = outputs['mask_camera']

        num_masks = src_masks.size(0)
        src_masks = src_masks.view(num_masks, 1, -1)
        target_masks = target_masks.view(num_masks, 1, -1)

        loss_dice = dice_loss(src_masks[mask], target_masks[mask], avg_factor.item(), mask_camera)
        loss_mask = sigmoid_ce_loss(src_masks[mask], target_masks[mask].float(), avg_factor.item(), mask_camera)
        loss_dice = loss_dice * self.loss_dice_weight
        loss_mask = loss_mask * self.loss_mask_weight
        
        losses['loss_dice'] = loss_dice
        losses['loss_mask'] = loss_mask

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_masks]

        indices: [(src_idx, tgt_idx)]
        """
        # [bs=1, num_query, num_classes]
        src_logits = outputs['pred_logits']
        # batch_idx, src_idx
        idx = self._get_src_permutation_idx(indices)
        # [bs, num_query]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J) * self.num_classes
            # set labels of track-appear slots to num_classes
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]].to(torch.int64)
            labels.append(labels_per_img)
        # [num_matched]
        target_classes_o = torch.cat(labels)
        # [bs, num_query]
        target_classes[idx] = target_classes_o
        label_weights = torch.ones_like(target_classes)
        # float tensor
        avg_factor = target_classes_o.numel()  # pos + mathced gt for disapper track
        avg_factor = reduce_mean(
                src_logits.new_tensor([avg_factor]))
        loss_ce = self.loss_cls(src_logits.flatten(0, 1), target_classes.flatten(0),
                                label_weights.flatten(0), avg_factor)
        loss_ce = loss_ce * self.loss_cls_weight
        losses = {'loss_cls': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses
    
    # @force_fp32(apply_to=('outputs'))
    def match_for_single_frame(self, outputs: dict, mask_camera, dec_lvl: int, if_step=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_masks_i = track_instances.pred_masks  # predicted masks of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_centers': track_instances.pred_centers.unsqueeze(0),
            'pred_masks': pred_masks_i.unsqueeze(0),
            'mask_camera': mask_camera,
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        # previously tracked, which is matched by rule
        matched_track_idxes = (track_instances.obj_idxes >= 0)
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        # new tgt indexes
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        # [num_untracked]
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, mask_camera):
            mask_preds, cls_preds = unmatched_outputs['pred_masks'], unmatched_outputs['pred_logits']
            bs, num_querys = mask_preds.shape[:2]
            # Also concat the target labels and masks
            targets = [untracked_gt_instances]
            if isinstance(targets[0], Instances):
                gt_labels = torch.cat([gt_per_img.labels for gt_per_img in targets])
                gt_masks = torch.cat([gt_per_img.masks for gt_per_img in targets])
            else:
                gt_labels = torch.cat([v["labels"] for v in targets])
                gt_masks = torch.cat([v["masks"] for v in targets])
            if len(gt_labels) == 0:
                return None
            mask_pred = mask_preds[0]
            cls_pred = cls_preds[0]
            src_idx, tgt_idx = self.matcher(mask_pred, cls_pred, gt_masks, gt_labels, mask_camera)
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            # [bs, num_pred, num_classes]
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            # [bs, num_pred, dim]
            'pred_masks': track_instances.pred_masks[unmatched_track_idxes].unsqueeze(0),
        }
        # [num_new_matched, 2]
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, mask_camera)

        # step5. update obj_idxes according to the new matching result.
        if new_matched_indices is not None:
            track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
            track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

            # step7. merge the unmatched pairs and the matched pairs.
            # [num_new_macthed + num_prev_mathed, 2]
            matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)
        else:
            matched_indices = prev_matched_indices

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device

        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])]
                                           )
            #print(self._current_frame_idx)
            self.losses_dict.update(
                {'frame_{}_{}_{}'.format(self._current_frame_idx, key, dec_lvl): value for key, value in new_track_loss.items()})
        #print('aux_outputs' in outputs)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_masks': aux_outputs['pred_masks'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, mask_camera)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        #print(if_step)
        if if_step:
            self._step()
        return track_instances
