import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.ops import roi_pool
import utils.boxes as box_utils

from core.config import cfg
from model.pcl.pcl import get_proposal_clusters, _get_highest_score_proposals, _get_multi_centers, get_proposal_clusters_semi
import nn as mynn
import utils.net as net_utils
from model.pcl.pcl import SLV
from core.label_helper import dynamic_func

import numpy as np

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, dim_out)
        self.bbox_pred = nn.Linear(dim_in, dim_out * 4)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred

def get_fast_rcnn_teacher_targets(boxes, gt_boxes, gt_boxes_classes, im_labels, history_label, cur_epoch):
    # if cur_epoch >= cfg.TRAIN.REWEIGHT_EPOCH and cfg.TRAIN.LABEL_UPDATE:

    #     # gt_scores = []
    #     # ratio = cfg.TRAIN.UPDATE_RATE
    #     # for h_s in history_label:
    #     #     cur_index = 1
    #     #     his_s = 1 * (1 - ratio) + h_s[cur_index] * ratio
    #     #     while cur_index < (len(h_s)-1):
    #     #         cur_index += 1
    #     #         his_s = his_s * (1 - ratio) + ratio * h_s[cur_index]
    #     #     gt_scores.append(his_s)
    #     gt_scores = [1 for i in history_label]

    #     proposals = {
    #         'gt_boxes': gt_boxes.data.cpu().numpy()[0],
    #         'gt_classes': np.expand_dims(gt_boxes_classes.data.cpu().numpy()[0, 0, :] + 1, -1),
    #         'gt_scores': np.expand_dims(np.asarray(gt_scores, np.float32), -1)
    #     }
    # else:
    #     gt_scores__ = [1 for i in history_label]
    #     # gt_scores__ = [1.0 if i[-1] != 0.0 else 0.0 for i in history_label]

    # if cfg.TRAIN.DYNAMIC_TYPE == "constant":
    #     constant_number = 5
    #     gt_scores__ = [float(i[constant_number] != 0) for i in history_label]
    # elif cfg.TRAIN.DYNAMIC_TYPE == "dynamic_linear":
    #     gt_scores__ = [1.0 if i[cur_epoch] != 0 else 0.0 for i in history_label]
    # elif cfg.TRAIN.DYNAMIC_TYPE == "dynamic_linear_weight":
    if cfg.TRAIN.LABEL_UPDATE:
        gt_scores__ = [i[-1] for i in history_label]
    else:
        gt_scores__ = [1 for i in history_label]

        
    proposals = {
        'gt_boxes': gt_boxes.data.cpu().numpy()[0],
        'gt_classes': np.expand_dims(gt_boxes_classes.data.cpu().numpy()[0, 0, :] + 1, -1),
        'gt_scores': np.expand_dims(np.asarray(gt_scores__, np.float32), -1)
    }

    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    return labels.reshape(-1).astype(np.int64).copy(), \
        cls_loss_weights.reshape(-1).astype(np.float32).copy(), \
        bbox_targets.astype(np.float32).copy(), \
        bbox_inside_weights.astype(np.float32).copy(), \
        bbox_outside_weights.astype(np.float32).copy(), \
        gt_assignment.astype(np.int64).copy()


def get_fast_rcnn_student_targets(boxes, gt_boxes_gpu, gt_boxes_classes, im_labels, history_label, cur_epoch, teacher_cls):
    teacher_boxes = np.zeros((0, 4), dtype=np.float32)
    teacher_classes = np.zeros((0, 1), dtype=np.int32)
    teacher_scores = np.zeros((0, 1), dtype=np.float32)
    num_images, num_classes = im_labels.shape
    teacher_score_thres = 0.5
    teacher_iou_thres = 0.3
    split_score_thres = 0.0
    match_iou_thres = 0.3
    begin_semi_epoch = 6

    gt_classes = np.asarray(np.expand_dims(gt_boxes_classes.data.cpu().numpy()[0, 0, :] + 1, -1), dtype=np.int32)
    gt_boxes = np.asarray(gt_boxes_gpu.data.cpu().numpy()[0], dtype=np.float32)

    # 用teacher的分数生成预测标签
    teacher_cls_20 = teacher_cls[:, 1:].clone().data.cpu().numpy()
    for i in xrange(num_classes):
        if im_labels[0, i] == 1:
            cls_prob_tmp = teacher_cls_20[:, i].copy()

            sort_arg = np.argsort(-cls_prob_tmp)
            sort_boxes = boxes[sort_arg]
            sort_cls_prob = cls_prob_tmp[sort_arg]

            for a in range(0, cls_prob_tmp.shape[0]):
                if sort_cls_prob[a] < teacher_score_thres and a != 0: break
                same_gt_idx = np.where(teacher_classes[:, 0] == (i+1))[0]
                insert_flag = False

                if same_gt_idx.shape[0] == 0 or a == 0:
                    insert_flag = True
                else:
                    overlaps = box_utils.bbox_overlaps(sort_boxes[a:a+1].astype(dtype=np.float32, copy=False), teacher_boxes[same_gt_idx])
                    if overlaps.max(axis=1)[0] < teacher_iou_thres:
                        insert_flag = True
                
                if insert_flag:
                    target_index = sort_arg[a]
                    teacher_boxes = np.vstack((teacher_boxes, boxes[target_index, :].reshape(1, -1)))
                    teacher_classes = np.vstack((teacher_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
                    teacher_scores = np.vstack((teacher_scores,
                                        cls_prob_tmp[target_index] * np.ones((1, 1), dtype=np.float32)))
                    teacher_cls_20[target_index, :] = 0

    # # 合并现有标签和teacher标签
    # fsod_boxes = np.zeros((0, 4), dtype=np.float32)
    # fsod_classes = np.zeros((0, 1), dtype=np.int32)
    # fsod_scores = np.zeros((0, 1), dtype=np.float32)
    # for i in range(gt_boxes.shape[0]):
    #     # 判断GT能否作为强监督标签加入到集合中
    #     if history_label[i][-1] > split_score_thres or cur_epoch < begin_semi_epoch:
    #         fsod_boxes = np.vstack((fsod_boxes, gt_boxes[i:i+1, :]))
    #         fsod_classes = np.vstack((fsod_classes, gt_classes[i:i+1, :]))
    #         fsod_scores = np.vstack((fsod_scores, float(history_label[i][-1].item() != 0 ) * np.ones((1, 1), dtype=np.float32)))
    
    # if sum([i[-1].item() for i in history_label]) > 0 or cur_epoch < begin_semi_epoch:
    #     pass
    # else:
    #     fsod_num = fsod_boxes.shape[0]
    #     for i in range(teacher_boxes.shape[0]):
    #         insert_flag = True
    #         for j in range(fsod_num):
    #             if fsod_classes[j, 0] == teacher_classes[i, 0]:
    #                 overlaps = box_utils.bbox_overlaps(teacher_boxes[i:i+1].astype(dtype=np.float32, copy=False), fsod_boxes[j:j+1, :])
    #                 if overlaps.max(axis=1)[0] > match_iou_thres:
    #                     insert_flag = False
    #                     break
            
    #         if insert_flag:
    #             fsod_boxes = np.vstack((fsod_boxes, teacher_boxes[i:i+1, :]))
    #             fsod_classes = np.vstack((fsod_classes, teacher_classes[i:i+1, :]))
    #             # fsod_scores = np.vstack((fsod_scores, teacher_scores[i:i+1]))
    #             fsod_scores = np.vstack((fsod_scores, np.ones((1, 1), dtype=np.float32)))
    
    proposals = {
        'gt_boxes': teacher_boxes,
        'gt_classes': teacher_classes,
        'gt_scores': teacher_scores
    }

    labels, cls_loss_weights, _, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())
    
    return labels.reshape(-1).astype(np.int64).copy(), \
           cls_loss_weights.reshape(-1).astype(np.float32).copy(), \
           bbox_targets.astype(np.float32).copy(), \
           bbox_inside_weights.astype(np.float32).copy(), \
           bbox_outside_weights.astype(np.float32).copy(), \
           _.astype(np.int64).copy(), \
           proposals


def get_fast_rcnn_targets(boxes, refine_score, im_labels, average=False):
    if average:
        cls_prob = sum(refine_score[1:]).data.cpu().numpy() / len(refine_score[1:])
    else:
        cls_prob = refine_score[-1].data.cpu().numpy()

    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps

    # proposals = _get_highest_score_proposals(boxes.copy(), cls_prob.copy(), im_labels.copy())
    if cfg.SUP.SWITCH and cfg.SUP.FASTER_MULTI_CENTERS:
        proposals = _get_multi_centers(boxes.copy(), cls_prob.copy(), im_labels.copy(), expand=cfg.SUP.FASTER_EXPAND, sub_cat=cfg.SUP.FASTER_SUB_CAT, remove=cfg.SUP.FASTER_REMOVE, merge=cfg.SUP.FASTER_MERGE, reweight=cfg.SUP.FASTER_REWEIGHT, fast=True)
    else:
        proposals = _get_highest_score_proposals(boxes.copy(), cls_prob.copy(), im_labels.copy())

    labels, cls_loss_weights, _, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    extra_dict = {}

    if "merge_scores" in proposals:
        extra_dict = proposals

    return labels.reshape(-1).astype(np.int64).copy(), \
           cls_loss_weights.reshape(-1).astype(np.float32).copy(), \
           bbox_targets.astype(np.float32).copy(), \
           bbox_inside_weights.astype(np.float32).copy(), \
           bbox_outside_weights.astype(np.float32).copy(), \
           extra_dict


def fast_rcnn_losses(cls_score, bbox_pred, labels, cls_loss_weights,
                     bbox_targets, bbox_inside_weights, bbox_outside_weights, im_labels=None):
    cls_loss = -(F.log_softmax(cls_score, dim=1)[range(cls_score.size(0)), labels].view(-1) * cls_loss_weights).mean()

    # print(im_labels.shape, labels.shape)

    bbox_loss = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_loss_weights)

    return cls_loss, bbox_loss


def fast_rcnn_losses_2(cls_score, bbox_pred, labels, cls_loss_weights,
                     bbox_targets, bbox_inside_weights, bbox_outside_weights, im_labels=None, gt_assignment=None, gt_box_num=None):
    origin_loss = -F.log_softmax(cls_score, dim=1)[range(cls_score.size(0)), labels].view(-1)
    cls_loss = (origin_loss * cls_loss_weights).mean()

    # print(cls_score.shape)
    logits = F.softmax(cls_score, 1)[range(cls_score.size(0)), labels].view(-1).detach()
    # print(logits.shape)

    cls_loss_per_anno = []
    for box_th_for_record in range(gt_box_num):
        append_loss = origin_loss[gt_assignment == box_th_for_record]
        if append_loss.shape[0] > 0:
            cls_loss_per_anno.append(append_loss.mean().item())
        else:
            cls_loss_per_anno.append(10.0)

    score_per_anno = []
    for box_th_for_record in range(gt_box_num):
        append_score = logits[gt_assignment == box_th_for_record]
        if append_score.shape[0] > 0:
            score_per_anno.append(append_score.max().item())
        else:
            score_per_anno.append(0.0)

    bbox_loss, reg_loss_per_anno = net_utils.smooth_l1_loss_2(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_loss_weights, gt_assignment=gt_assignment, gt_box_num=gt_box_num)

    assert len(cls_loss_per_anno) == len(reg_loss_per_anno)

    total_loss_per_anno = []
    for per_anno in range(len(reg_loss_per_anno)):
        total_loss_per_anno.append(cls_loss_per_anno[per_anno] + reg_loss_per_anno[per_anno])

    extra_dict = {
        "loss": total_loss_per_anno,
        "score": score_per_anno
    }

    return cls_loss, bbox_loss, extra_dict


@torch.no_grad()
def foward_net_M(net, proposals):
    rois = torch.from_numpy(proposals["gt_boxes"]).cuda().float()
    M_rois_feat = roi_pool(net.final_conv, [rois], [7, 7], net.Conv_Body.spatial_scale)
    M_rois_feat, roi_feature_clone = net.Box_Head(M_rois_feat, rois)
    return net.FRCNN_Outs(M_rois_feat)


def merge_state2weight(cur_epoch, proposals, pseudo_gt_cls):
    pseudo_gt_scores = proposals["gt_scores"]
    pseudo_gt_assignment = proposals["gt_classes"][:, 0]

    forward_scores = pseudo_gt_cls[np.arange(pseudo_gt_cls.shape[0]), pseudo_gt_assignment]
    adjust_scores = pseudo_gt_scores * forward_scores[:, np.newaxis]
    if not (adjust_scores == 0).all():
        print("find {0} box".format(np.where(adjust_scores[:, 0] != 0)[0].shape[0]))
    return adjust_scores
    


@torch.no_grad()
def get_fast_rcnn_slv_targets(rois_, source_score, target, img_shape, SLV_M_THRES, SLV_prior_thres, fg_iou_th, gamma_param_list, net, cur_step):

    proposals = SLV(rois_, source_score, target, img_shape, SLV_M_THRES, SLV_prior_thres, fg_iou_th, gamma_param_list)

    if cfg.SLV.degree_add:
        pseudo_gt_cls, pseudo_gt_det = foward_net_M(net, proposals)
        proposals["gt_scores"] = merge_state2weight(cur_step, proposals, F.softmax(pseudo_gt_cls, dim=-1).detach().cpu().numpy())

    labels, cls_loss_weights, _, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(rois_.detach().cpu().numpy(), proposals, target.detach().cpu().numpy())

    return labels.reshape(-1).astype(np.int64).copy(), \
           cls_loss_weights.reshape(-1).astype(np.float32).copy(), \
           bbox_targets.astype(np.float32).copy(), \
           bbox_inside_weights.astype(np.float32).copy(), \
           bbox_outside_weights.astype(np.float32).copy()




