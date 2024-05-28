# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
from .pos_neg_select import select_pos_neg

import pdb 

class CondInst_segm(nn.Module):
    def __init__(self, detector, rel_coord=True, freeze_detr=False):
        super().__init__()
        self.detector = detector
        """
        self.detector.keys() : dict_keys(['backbone', 'sem_seg_head', 'criterion', 'num_queries', 'object_mask_threshold', 'overlap_threshold', 
                               'metadata', 'size_divisibility', 'sem_seg_postprocess_before_inference', 'pixel_mean', 'pixel_std', 'semantic_on','instance_on', 'panoptic_on', 'test_topk_per_image', 'data_loader', 'focus_on_box', 'transform_eval', 'pano_temp', 'semantic_ce_loss'])
        """
        self.detector["backbone"] = self.detector["backbone"].cuda()
        self.detector["sem_seg_head"] = self.detector["sem_seg_head"].cuda()

        hidden_dim = 256 #  maskdino feature dimension

        self.reid_embed_head = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward(self, samples, det_targets, ref_targets,ref_targets_valid,  criterion, train=False):
        """
        Forward pass

        Args:
            samples: batch of images
            det_targets: detection targets
            ref_targets: reference targets
            criterion: loss function
            train: boolean indicating whether to train or evaluate

        Returns:
            outputs: dictionary containing class logits, box coordinates, mask predictions, and re-identification embeddings
        """
        #pdb.set_trace()
        image_sizes = samples.image_sizes # 64 x 1024 x 1024 |type: <class 'detectron2.structures.image_list.ImageList'>

        # Extract height and width from the tuple
        height, width = image_sizes[0] # 1024 , 1024 | multiple size by default in IDOL;  #len(det_targets), len(ref_targets): # 32, 32
       
        #create reference and keyframe indices
        bz = len(samples)//2
        key_ids = list(range(0,bz*2-1,2)) #print(key_ids) #0,2,4,6,8 <even numbers>
        ref_ids = list(range(1,bz*2,2))   #print(ref_ids) #1,3,5,7,9 <odd  numbers>

        det_samples = samples.tensor[key_ids]
        ref_samples = samples.tensor[ref_ids]

        det_features = self.detector["backbone"](det_samples) # dict_keys(['res2', 'res3', 'res4', 'res5'])
        ref_features = self.detector["backbone"](ref_samples)

        
        """dec_layers = 9, Each produces 300 queries: 9x300x256 [Important]"""
        det_outputs,det_mask_dict = self.detector["sem_seg_head"](det_features,targets=det_targets) #dict, dict
        det_losses = self.detector["criterion"](det_outputs, det_targets, det_mask_dict)            # bipartite matching-based loss

        for k in list(det_losses.keys()):
            if k in  self.detector["criterion"].weight_dict:
                det_losses[k] *=  self.detector["criterion"].weight_dict[k]

        #pdb.set_trace()

      
        ref_outputs,ref_mask_dict = self.detector["sem_seg_head"](ref_features,targets=None)
        # print(ref_outputs.keys(),ref_mask_dict.keys())

        #print(ref_losses)
        #ref_outputs:dict_keys(['pred_features' <torch.Size([2, 399, 256])>, 'pred_logits' <torch.Size([2, 300, 7])>, 
        #                       'pred_masks' <torch.Size([2, 300, 256, 256])>, 'pred_boxes'<torch.Size([2, 300, 4])>, 
        #                       'aux_outputs', 'interm_outputs']) 
        
        #ref_mask_dict:dict_keys(['known_indice', 'batch_idx', 'map_known_indice', 'known_lbs_bboxes', 'know_idx', 'pad_size',
        #                        'scalar', 'output_known_lbs_bboxes'])

        # print("det_outputs['pred_features'].shape", det_outputs['pred_features'].shape)
        # print("det_outputs['pred_logits'].shape", det_outputs['pred_logits'].shape)
        # print("det_outputs['pred_boxes'].shape", det_outputs['pred_boxes'].shape)
        # print("det_outputs['pred_masks'].shape", det_outputs['pred_masks'].shape)

        outputs_layer = {'pred_logits': det_outputs['pred_logits'], 'pred_boxes': det_outputs['pred_boxes']}

        # for training & log evaluation loss
        indices, matched_ids = criterion.matcher(outputs_layer, det_targets)

        #print("contrastive")
        ref_cls = ref_outputs['pred_logits'].sigmoid()
        #print("ref_outputs['pred_logits']", ref_cls)
        contrast_items = select_pos_neg(ref_outputs['pred_boxes'], matched_ids, ref_targets, det_targets, self.reid_embed_head, det_outputs['pred_features'], ref_outputs['pred_features'], ref_cls)
        #contrast_items = select_pos_neg(inter_references_ref[-1], matched_ids, ref_targets, det_targets, self.reid_embed_head, hs[-1], hs_ref[-1], ref_cls)

        #print("contrastive_items")

        outputs = {}
        indices_list=[]

        indices_list.append(indices)
        # outputs['pred_samples'] = inter_samples[-1]
        outputs['pred_logits'] = det_outputs['pred_logits']
        outputs['pred_boxes']  = det_outputs['pred_boxes']
        outputs['pred_masks']  = det_outputs['pred_masks']
        outputs['pred_qd']     = contrast_items

        #pdb.set_trace()
        # if self.detr.aux_loss:
        #     outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
        # print("values:", outputs['pred_logits'][0].shape,  len(outputs['pred_logits']), outputs['pred_boxes'][0].shape,len(outputs['pred_boxes']), outputs['pred_masks'][0].shape, len(outputs['pred_masks']))
        
        if train:
            loss_dict = criterion(outputs, det_targets, ref_targets, indices_list)
        else:
            loss_dict = None
        loss_dict = {**loss_dict, **det_losses} # merge losses

        #print("loss_dict", loss_dict)
        return outputs, loss_dict


    def inference_forward(self, samples, size_divisib=32):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=size_divisib)
    
        features, pos = self.detr.backbone(samples)
        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, feat in enumerate(features[1:]):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos[l+1])
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        query_embeds = self.detr.query_embed.weight

    
        hs, memory, init_reference, inter_references, inter_samples, enc_outputs_class, enc_outputs_coord_unact = \
            self.detr.transformer(srcs, masks, poses, query_embeds)
        

        outputs = {}

        reference = inter_references[-1 - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = self.detr.class_embed[-1](hs[-1])
        tmp = self.detr.bbox_embed[-1](hs[-1])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs['pred_logits'] = outputs_class
        outputs['pred_boxes'] = outputs_coord
        inst_embed = self.reid_embed_head(hs[-1])
        outputs['pred_inst_embed'] = inst_embed


        outputs['reference_points'] = inter_references[-2, :, :, :2]
        dynamic_mask_head_params = self.controller(hs[-1])    # [bs, num_quries, num_params]            
        bs, num_queries, _ = dynamic_mask_head_params.shape
        num_insts = [num_queries for i in range(bs)]
        reference_points = []
        for i, image_size_i in enumerate(image_sizes):
            orig_h, orig_w = image_size_i
            orig_h = torch.as_tensor(orig_h).to(outputs['reference_points'][i])
            orig_w = torch.as_tensor(orig_w).to(outputs['reference_points'][i])
            scale_f = torch.stack([orig_w, orig_h], dim=0)
            ref_cur_f = outputs['reference_points'][i] * scale_f[None, :]
            reference_points.append(ref_cur_f.unsqueeze(0))
        # reference_points: [1, N * num_queries, 2]
        # mask_head_params: [1, N * num_queries, num_params]
        reference_points = torch.cat(reference_points, dim=1)
        mask_head_params = dynamic_mask_head_params.reshape(1, -1, dynamic_mask_head_params.shape[-1])
        outputs = self.forward_mask_head_train(outputs, memory, spatial_shapes, 
                                                reference_points, mask_head_params, num_insts)
        # outputs['pred_masks']: [bs, num_queries, num_frames, H/4, W/4]
        outputs['pred_masks'] = torch.cat(outputs['pred_masks'], dim=0)
     

        return outputs


    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts):
        bs, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, spatial_indx: spatial_indx + 1 * h * w, :].reshape(bs, 1, h, w, c).permute(0,4,1,2,3)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w
        
        pred_masks = []
        for iframe in range(1):
            encod_feat_f = []
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]

            # feats = [] # features[3], features[2], features[1]
            # for i in range(self.detr.num_feature_levels - 1, 0, -1):
            #     N, _c, _h, _w = features[i].tensors.shape
            #     feats.append(features[i].tensors.reshape(bs, self.detr.num_frames, _c, _h, _w)[:,iframe,:,:,:])
            
            decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
          
            ######### conv ##########
            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points, mask_head_params, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord )
            # mask_logits: [1, num_queries_all, H/4, W/4]

            # mask_f = mask_logits.unsqueeze(2).reshape(bs, nq, 1, decod_feat_f.shape[-2], decod_feat_f.shape[-1])  # [bs, selected_queries, 1, H/4, W/4]
            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)  
        
        # outputs['pred_masks'] = torch.cat(pred_masks, 2) # [bs, selected_queries, num_frames, H/4, W/4]
        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))
        
        outputs['pred_masks'] = output_pred_masks
        return outputs

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]



def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)





def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


from detectron2.structures import Instances

def segmentation_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
    ):

    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        mask = F.interpolate(results.pred_masks.float(), size=(output_height, output_width), mode='nearest')
        mask = mask.squeeze(1).byte()
        results.pred_masks = mask

    return results
