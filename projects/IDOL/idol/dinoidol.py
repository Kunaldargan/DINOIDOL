# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from fvcore.nn import giou_loss, smooth_l1_loss

from .models.deformable_detr import SetCriterion
from .models.matcher import HungarianMatcher

from .models.segmentation_condInst_dino import CondInst_segm, segmentation_postprocess

from detectron2.projects.maskdino import MaskDINO

from .models.tracker import IDOL_Tracker
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
from .data.coco import convert_coco_poly_to_mask
import torchvision.ops as ops

import pdb
__all__ = ["DINOIDOL"]


@META_ARCH_REGISTRY.register()
class DINOIDOL(nn.Module):
    """
    Implement DINOIDOL
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.clip_stride = cfg.MODEL.IDOL.CLIP_STRIDE

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.IDOL.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IDOL.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IDOL.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.IDOL.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.IDOL.INFERENCE_FW
        self.inference_tw = cfg.MODEL.IDOL.INFERENCE_TW
        self.memory_len = cfg.MODEL.IDOL.MEMORY_LEN
        self.nms_pre = cfg.MODEL.IDOL.NMS_PRE
        self.add_new_score = cfg.MODEL.IDOL.ADD_NEW_SCORE 
        self.batch_infer_len = cfg.MODEL.IDOL.BATCH_INFER_LEN


        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.IDOL.NUM_CLASSES
        self.mask_stride = cfg.MODEL.IDOL.MASK_STRIDE
        self.match_stride = cfg.MODEL.IDOL.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON

        self.coco_pretrain = cfg.INPUT.COCO_PRETRAIN
        hidden_dim = cfg.MODEL.IDOL.HIDDEN_DIM
        num_queries = cfg.MODEL.IDOL.NUM_OBJECT_QUERIES


        # Loss parameters:
        mask_weight = cfg.MODEL.IDOL.MASK_WEIGHT
        dice_weight = cfg.MODEL.IDOL.DICE_WEIGHT
        giou_weight = cfg.MODEL.IDOL.GIOU_WEIGHT
        l1_weight = cfg.MODEL.IDOL.L1_WEIGHT
        class_weight = cfg.MODEL.IDOL.CLASS_WEIGHT
        reid_weight = cfg.MODEL.IDOL.REID_WEIGHT
        deep_supervision = cfg.MODEL.IDOL.DEEP_SUPERVISION

        focal_alpha = cfg.MODEL.IDOL.FOCAL_ALPHA

        set_cost_class = cfg.MODEL.IDOL.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.IDOL.SET_COST_BOX
        set_cost_giou = cfg.MODEL.IDOL.SET_COST_GIOU

        N_steps = hidden_dim // 2

        model = MaskDINO.from_config(cfg) #create maskdino model based on joint config
        
        self.detr = CondInst_segm(model, freeze_detr=False, rel_coord=True )

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(multi_frame=True, # True, False
                                    cost_class=set_cost_class,
                                    cost_bbox=set_cost_bbox,
                                    cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight}
        weight_dict["loss_reid"] = reid_weight
        weight_dict["loss_reid_aux"] = reid_weight*1.5
        weight_dict["loss_mask"] = mask_weight
        weight_dict["loss_dice"] = dice_weight


        """
        Deep supervision is a technique used in deep learning, particularly in object detection and segmentation tasks. 
        It involves adding auxiliary losses at intermediate layers of the network, in addition to the main loss function at
        the output layer. The goal of deep supervision is to provide additional guidance to the network during training, 
        helping it to learn more robust and accurate representations. 
        """
        # if deep_supervision:
        #     aux_weight_dict = {}
        #     for i in range(dec_layers - 1):
        #         aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        #     weight_dict.update(aux_weight_dict)
  
        losses = ['labels', 'boxes', 'reid'] #'labels', 'boxes', 'masks','reid']
        
        self.criterion = SetCriterion(self.num_classes, matcher, weight_dict, losses, 
                             mask_out_stride=self.mask_stride,
                             focal_alpha=focal_alpha,
                             num_frames = self.num_frames)

        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.merge_device = "cpu" if self.merge_on_cpu else self.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other./datasets/EV18/train/JPEGImages/seq6/frame081.png information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            #pdb.set_trace()
            images = self.preprocess_image(batched_inputs) #images shape : C x 3 x 1024 x 1024 
            gt_instances = []
            for video in batched_inputs:
                for j, inst in enumerate(video["instances"]):
                    frame, frame_idx = inst
                    #print(frame_idx)
                    #print(video["file_names"][j])
                    gt_instances.append(frame.to(self.device))

            det_targets,ref_targets, ref_targets_valid = self.prepare_targets(gt_instances)
            output, loss_dict = self.detr(images, det_targets,ref_targets, ref_targets_valid, self.criterion, train=True)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        elif self.coco_pretrain:  #evluate during coco pretrain
            images = self.preprocess_coco_image(batched_inputs)
            output = self.detr.inference_forward(images, size_divisib=32) #
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.coco_inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = segmentation_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            images = self.preprocess_image(batched_inputs)
            video_len = len(batched_inputs[0]['file_names'])
            clip_length = self.batch_infer_len
            #split long video into clips to form a batch input 
            if video_len > clip_length:
                num_clips = math.ceil(video_len/clip_length)
                logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
                for c in range(num_clips):
                    start_idx = c*clip_length
                    end_idx = (c+1)*clip_length
                    clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                    clip_images = self.preprocess_image(clip_inputs)
                    clip_output = self.detr.inference_forward(clip_images)
                    logits_list.append(clip_output['pred_logits'])
                    boxes_list.append(clip_output['pred_boxes'])
                    embed_list.append(clip_output['pred_inst_embed'])
                    # points_list.append(clip_output['reference_points'])
                    masks_list.append(clip_output['pred_masks'].to(self.merge_device))
                output = {
                    'pred_logits':torch.cat(logits_list,dim=0),
                    'pred_boxes':torch.cat(boxes_list,dim=0),
                    'pred_inst_embed':torch.cat(embed_list,dim=0),
                    # 'reference_points':torch.cat(points_list,dim=0),
                    'pred_masks':torch.cat(masks_list,dim=0),
                }    
            else:
                images = self.preprocess_image(batched_inputs)
                output = self.detr.inference_forward(images)
            idol_tracker = IDOL_Tracker(
                    init_score_thr= 0.2,
                    obj_score_thr=0.1,
                    nms_thr_pre=self.nms_pre,  #0.5
                    nms_thr_post=0.05,
                    addnew_score_thr = self.add_new_score, #0.2
                    memo_tracklet_frames = 10,
                    memo_momentum = 0.8,
                    long_match = self.inference_tw,
                    frame_weight = (self.inference_tw|self.inference_fw),
                    temporal_weight = self.inference_tw,
                    memory_len = self.memory_len
                    )
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            video_output = self.inference(output, idol_tracker, (height, width), images.image_sizes[0])  # (height, width) is resized size,images. image_sizes[0] is original size

            return video_output


    def prepare_targets(self, targets):
        #pdb.set_trace()
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            gt_masks = targets_per_image.gt_masks.tensor
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
            #print("inst_ids", inst_ids)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, 'inst_id':inst_ids, 'valid':valid_id})
        bz = len(new_targets)//2
        key_ids = list(range(0,bz*2-1,2))
        ref_ids = list(range(1,bz*2,2))
        det_targets = [new_targets[_i] for _i in key_ids]
        ref_targets = [new_targets[_i] for _i in ref_ids]
        ref_targets_valids = [new_targets[_i] for _i in ref_ids]
        #print("det_targets1:",det_targets)

        for i in range(bz):  # fliter empty object in key frame
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            ref_targets_valid = ref_targets_valids[i]
            
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    ref_target[k] = v[valid_i]

            if False in ref_targets_valid['valid']:
                valid_i = ref_targets_valid['valid'].clone()
                for k,v in ref_targets_valid.items():
                    ref_targets_valid[k] = v[valid_i]


        #print("det_targets2:",det_targets)
        return det_targets,ref_targets, ref_targets_valids

    def inference(self, outputs, tracker, ori_size, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        video_dict = {}
        vido_logits = outputs['pred_logits']
        video_output_masks = outputs['pred_masks']
        output_h, output_w = video_output_masks.shape[-2:]
        video_output_boxes = outputs['pred_boxes']
        video_output_embeds = outputs['pred_inst_embed']
        vid_len = len(vido_logits)
        for i_frame, (logits, output_mask, output_boxes, output_embed) in enumerate(zip(
            vido_logits, video_output_masks, video_output_boxes, video_output_embeds
         )):
            scores = logits.sigmoid().cpu().detach()  #[300,42]
            max_score, _ = torch.max(logits.sigmoid(),1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            box_score = torch.max(logits.sigmoid()[indices],1)[0]
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)
            det_labels = torch.argmax(logits.sigmoid()[indices],dim=1)
            track_feats = output_embed[indices]
            det_masks = output_mask[indices]
            bboxes, labels, ids, indices = tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            masks = det_masks,
            track_feats=track_feats,
            frame_id=i_frame,
            indices = indices)
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            for query_i, id in zip(indices,ids):
                if id in video_dict.keys():
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1
                else:
                    video_dict[id] = {
                        'masks':[None for fi in range(i_frame)], 
                        'boxes':[None for fi in range(i_frame)], 
                        'scores':[None for fi in range(i_frame)], 
                        'valid':0}
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1

            for k,v in video_dict.items():
                if len(v['masks'])<i_frame+1: #padding None for unmatched ID
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['boxes'].append(None)
            check_len = [len(v['masks']) for k,v in video_dict.items()]
            # print('check_len',check_len)

            #  filtering sequences that are too short in video_dict (noise)，the rule is: if the first two frames are None and valid is less than 3
            if i_frame>8:
                del_list = []
                for k,v in video_dict.items():
                    if v['masks'][-1] is None and  v['masks'][-2] is None and v['valid']<3:
                        del_list.append(k)   
                for del_k in del_list:
                    video_dict.pop(del_k)                      

        del outputs
        logits_list = []
        masks_list = []

        for inst_id,m in  enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            scores_temporal = []
            for k in score_list_ori:
                if k is not None:
                    scores_temporal.append(k)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            else:
                print('non valid temporal_score_type')
                import sys;sys.exit(0)
            logits_list.append(logits_i)
            
            # category_id = np.argmax(logits_i.mean(0))
            masks_list_i = []
            for n in range(vid_len):
                mask_i = video_dict[m]['masks'][n]
                if mask_i is None:    
                    zero_mask = None # padding None instead of zero mask to save memory
                    masks_list_i.append(zero_mask)
                else:
                    pred_mask_i =F.interpolate(mask_i[:,None,:,:],  size=(output_h*4, output_w*4) ,mode="bilinear", align_corners=False).sigmoid()
                    pred_mask_i = pred_mask_i[:,:,:image_sizes[0],:image_sizes[1]] #crop the padding area
                    pred_mask_i = (F.interpolate(pred_mask_i, size=(ori_size[0], ori_size[1]), mode='nearest')>0.5)[0,0].cpu() # resize to ori video size
                    masks_list_i.append(pred_mask_i)
            masks_list.append(masks_list_i)
        if len(logits_list)>0:
            pred_cls = torch.stack(logits_list)
        else:
            pred_cls = []

        if len(pred_cls) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]
                labels = is_above_thres[1]
                out_masks = [masks_list[valid_id] for valid_id in is_above_thres[0]]
            else:
                scores, labels = pred_cls.max(-1)
                out_masks = masks_list
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_masks = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        # print(len(batched_inputs))
        # print(type(batched_inputs))
        # print(type(batched_inputs[0]))
        # print(batched_inputs[0].keys())

        # print("height:",   batched_inputs[0]["height"])
        # print("width:",    batched_inputs[0]["width"])
        # print("length:",   batched_inputs[0]["length"])
        # print("video_id:", batched_inputs[0]["video_id"])
        # print("image:",    type(batched_inputs[0]["image"]))
        # print("image:",    len(batched_inputs[0]["image"]))
        # print("image:",    batched_inputs[0]["image"][0].shape)
        #pdb.set_trace()

        # for i in range(len(batched_inputs)):
        #     print("file:", batched_inputs[i]["file_names"])
        #    # if batched_inputs[i]["file_names"] == './datasets/EV18/train/JPEGImages/seq16/frame093.png':
        #    #     pdb.set_trace()

        # print("batched_inputs[0]", batched_inputs[0])

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        
       # image[0] shape : (3,1024,1024)

        images = ImageList.from_tensors(images)
        return images


    def coco_inference(self, box_cls, box_pred, mask_pred, image_sizes):
      
        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, image_size) in enumerate(zip(
            box_cls, box_pred, image_sizes
        )):



            prob = logits_per_image.sigmoid()
            nms_scores,idxs = torch.max(prob,1)
            boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
            keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
            prob = prob[keep_indices]
            box_pred_per_image = box_pred_per_image[keep_indices]
            mask_pred_i = mask_pred[i][keep_indices]

            topk_values, topk_indexes = torch.topk(prob.view(-1), 100, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
            # topk_boxes = topk_indexes // logits_per_image.shape[1]
            labels = topk_indexes % logits_per_image.shape[1]
            scores_per_image = scores
            labels_per_image = labels

            box_pred_per_image = box_pred_per_image[topk_boxes]
            mask_pred_i = mask_pred_i[topk_boxes]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                N, C, H, W = mask_pred_i.shape
                mask = F.interpolate(mask_pred_i, size=(H*4, W*4), mode='bilinear', align_corners=False)
                mask = mask.sigmoid() > 0.5
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results



    def preprocess_coco_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images



    