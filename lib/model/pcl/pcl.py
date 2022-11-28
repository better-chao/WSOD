from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.net as net_utils
from core.config import cfg

import numpy as np
from numba import jit, vectorize, float32
from sklearn.cluster import KMeans
import scipy.stats as st
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def PCL(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(),
        im_labels.copy())

    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy()}


def OICR(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps


    if cfg.SUP.OICR_MULTI_CENTERS and cfg.SUP.SWITCH:
        proposals = _get_multi_centers(boxes, cls_prob, im_labels, expand=cfg.SUP.OICR_EXPAND, sub_cat=cfg.SUP.OICR_SUB_CAT, remove=cfg.SUP.OICR_REMOVE, merge=cfg.SUP.OICR_MERGE, reweight=cfg.SUP.OICR_REWEIGHT)
    else:
        # OICR自带基础baseline
        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels)
        
        
    # proposals = _get_multi_centers(boxes, cls_prob, im_labels)

    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy()}


def SLV(proposals, source_score, target, img_shape, SLV_M_THRES, SLV_prior_thres, fg_iou_th, gamma_param_list):
    # print(proposals.shape, source_score.shape, target, img_shape)
    gt_boxes = torch.zeros((0, 4), dtype=torch.float32).cuda()
    gt_classes = torch.zeros((0, 1), dtype=torch.int32).cuda()
    gt_scores = torch.zeros((0, 1), dtype=torch.float32).cuda()
    gt_P = np.zeros((0, 1), dtype=np.float32)

    c_score = source_score[:, 1:].clone() if source_score.shape[-1] == 21 else source_score.clone()

    for _i_ in range(target.shape[-1]):
        if target[0, _i_] == 1:
            paramter = gamma_param_list[_i_]
            cls_prob_tmp = c_score[:, _i_]
            valid_inds = torch.where(cls_prob_tmp > 0.001)[0]
            
            val_proposals = proposals[valid_inds]
            val_proposals_scores = cls_prob_tmp[valid_inds]

            # 如果没有满足条件的框
            if val_proposals_scores.shape[0] == 0: continue

            M = torch.zeros(img_shape).cuda()
            for p in range(val_proposals.shape[0]):
                int_p = val_proposals[p].int()
                if int_p[2] <= int_p[0] or int_p[3] <= int_p[1]: continue
                M[int_p[1]:int_p[3], int_p[0]:int_p[2]] += val_proposals_scores[p]
            
            M = (M - M.min()) / (M.max() - M.min())
            M_binary = (M > SLV_M_THRES[_i_]).byte().cpu().numpy()
    
            contours, hierarchy = cv2.findContours(M_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # if len(contours) != 1: continue
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                P = st.gamma.pdf(w / h, paramter[0], scale=paramter[2])
                if P >= SLV_prior_thres[_i_]:
                    target_bbox = torch.from_numpy(np.asarray([[x, y, x+w, y+h]])).float().cuda()
                    gt_boxes = torch.cat((gt_boxes, target_bbox), dim=0)
                    gt_classes = torch.cat((gt_classes, (_i_ + 1) * torch.ones((1, 1), dtype=torch.int32).cuda()), dim=0)
                    gt_scores = torch.cat((gt_scores, torch.ones((1, 1), dtype=torch.float32).cuda()), dim=0)

                    gt_P = np.vstack([gt_P, P * np.ones([1, 1], dtype=np.float32)])

    if gt_boxes.shape[0] != 0:
        ious = box_utils.bbox_iou(gt_boxes, proposals)
        max_overlap = ious.max(axis=0)[0]
        gt_assignment = ious.argmax(axis=0)

        pseudo_labels = gt_classes[gt_assignment, 0].clone().detach()
        loss_weights = gt_scores[gt_assignment, 0].clone().detach()

        bg_inds = max_overlap.lt(fg_iou_th).nonzero(as_tuple=False)[:, 0]
        pseudo_labels[bg_inds] = 0

        loss_weights[torch.where(max_overlap < 0.1)[0]] = 0.0
        
        return {
            'cls_loss_weights': loss_weights.detach().cpu().numpy().astype(np.float32).reshape(1, -1),
            'labels': pseudo_labels.detach().cpu().numpy().astype(np.int64).reshape(1, -1),
            'gt_assignment': None,
            'gt_boxes': gt_boxes.detach().cpu().numpy(),
            'gt_classes': gt_classes.detach().cpu().numpy(),
            'gt_scores': gt_scores.detach().cpu().numpy(),
            'gt_P': gt_P
        }
    else:
        return {
            'cls_loss_weights': np.zeros(proposals.shape[0]).astype(np.float32).reshape(1, -1),
            'labels': np.zeros(proposals.shape[0]).astype(np.int64).reshape(1, -1),
            'gt_assignment': None,
            'gt_boxes': np.asarray([[0, 0, 10, 10]]).astype(np.float32),
            'gt_classes': np.zeros([1, 1]).astype(np.int64),
            'gt_scores': np.zeros([1, 1]).astype(np.float32),
            'gt_P': np.zeros([1, 1]).astype(np.float32)
        }



def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals

def merge_objectness(boxes, target_box):
    max_x = int(max(np.max(boxes[:, 2]), target_box[2]))
    min_x = int(min(np.min(boxes[:, 0]), target_box[0]))
    max_y = int(max(np.max(boxes[:, 3]), target_box[3]))
    min_y = int(min(np.min(boxes[:, 1]), target_box[1]))
    x_emulator = np.zeros([max_x, ])
    y_emulator = np.zeros([max_y, ])
    for b in range(boxes.shape[0]):
        x_emulator[max(int(boxes[b, 0]-1), 0):min(int(boxes[b, 2]-1), max_x)] = 1
        y_emulator[max(int(boxes[b, 1]-1), 0):min(int(boxes[b, 3]-1), max_y)] = 1
    x_objectness = np.sum(x_emulator) / (max_x - min_x)
    y_objectness = np.sum(y_emulator) / (max_y - min_y)
    return x_objectness * y_objectness


# 自己看一下，完成基础标签生成、移除、扩展整个优化
def _get_multi_centers(boxes, cls_prob, im_labels, expand=False, sub_cat=False, remove=False, merge=False, reweight=False, fast=False):
    seed_score_thres = [0.5 for _ in range(20)]
    min_iou_thres = 0.2
    
    expand_score_thres = [0.5 for _ in range(20)]
    
    expand_iou_thres = 0.3
    contian_thres = 0.85
    contain_min_num = 2
    objectness_thres = 0.64

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)

    expand_gt_boxes = np.zeros((0, 4), dtype=np.float32)
    expand_gt_classes = np.zeros((0, 1), dtype=np.int32)
    expand_gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            sort_arg = np.argsort(-cls_prob_tmp)

            sort_boxes = boxes[sort_arg]
            sort_cls_prob = cls_prob_tmp[sort_arg]

            max_index = sort_arg[0]

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0

            append_gt_list = []


            for a in range(1, cls_prob_tmp.shape[0]):
                if sort_cls_prob[a] < seed_score_thres[i]: break

                overlaps = box_utils.bbox_overlaps(sort_boxes[a:a+1].astype(dtype=np.float32, copy=False), 
                gt_boxes[np.where(gt_classes[:, 0] == (i+1))[0]])

                if overlaps.max(axis=1)[0] <= min_iou_thres:
                    target_index = sort_arg[a]
                    
                    # remove part
                    cur_class_gt_box = gt_boxes[np.where(gt_classes[:, 0] == (i+1))[0]]
                    cur_class_cur_contain_matrix = box_utils.bbox_contain(cur_class_gt_box,sort_boxes[a:a+1].astype(dtype=np.float32, copy=False))[:,0]

                    if remove and cur_class_cur_contain_matrix.max() >= contian_thres: continue
                    
                    # gt vstack
                    gt_boxes = np.vstack((gt_boxes, boxes[target_index, :].reshape(1, -1)))
                    gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
                    gt_scores = np.vstack((gt_scores,
                                        cls_prob_tmp[target_index] * np.ones((1, 1), dtype=np.float32)))
                    cls_prob[target_index, :] = 0
                    append_gt_list.append(sort_arg[a])

            if expand:
                for a in range(1, cls_prob_tmp.shape[0]):
                    # if sort_arg[a] in append_gt_list: continue
                    if sort_cls_prob[a] <= expand_score_thres[i]: break
                    single_class_gt_box = gt_boxes[np.where(gt_classes[:, 0] == (i+1))[0]]
                    contains_matrix = box_utils.bbox_contain(sort_boxes[a:a+1].astype(dtype=np.float32, copy=False),
                    single_class_gt_box)[0]

                    contains_num = np.sum(contains_matrix > contian_thres)

                    if contains_num >= contain_min_num:
                        target_index = sort_arg[a]
                        contains_gt_boxes = single_class_gt_box[np.where(contains_matrix > contian_thres)[0], :]
                        objectness = merge_objectness(contains_gt_boxes, boxes[target_index, :])
                        if objectness < objectness_thres: continue

                        if expand_gt_boxes.shape[0] == 0:
                            expand_gt_boxes = np.vstack((expand_gt_boxes, boxes[target_index, :].reshape(1, -1)))
                            expand_gt_classes = np.vstack((expand_gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
                            if not reweight:
                                expand_gt_scores = np.vstack((expand_gt_scores,
                                                    cls_prob_tmp[target_index] * np.ones((1, 1), dtype=np.float32)))
                            else:
                                max_score = (gt_scores[np.where(gt_classes[:, 0] == (i+1))[0]][contains_matrix > contian_thres]).max()
                                expand_gt_scores = np.vstack((expand_gt_scores, max_score * np.ones((1, 1), dtype=np.float32)))

                            cls_prob[target_index, :] = 0
                        else:
                            expand_overlaps =box_utils.bbox_overlaps(sort_boxes[a:a+1].astype(dtype=np.float32, copy=False),
                            expand_gt_boxes)

                            if expand_overlaps.max(axis=1)[0] < expand_iou_thres:
                                expand_gt_boxes = np.vstack((expand_gt_boxes, boxes[target_index, :].reshape(1, -1)))
                                expand_gt_classes = np.vstack((expand_gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
                                if not reweight:
                                    expand_gt_scores = np.vstack((expand_gt_scores,
                                                        cls_prob_tmp[target_index] * np.ones((1, 1), dtype=np.float32)))
                                else:
                                    max_score = (gt_scores[np.where(gt_classes[:, 0] == (i+1))[0]][contains_matrix > contian_thres]).max()
                                    expand_gt_scores = np.vstack((expand_gt_scores, max_score * np.ones((1, 1), dtype=np.float32)))

                                cls_prob[target_index, :] = 0

    merge_boxes, merge_classes, merge_scores = None, None, None

    if expand_gt_boxes.shape[0] != 0:
        merge_filter = np.full(gt_boxes.shape[0], False)
        for ex in range(expand_gt_boxes.shape[0]):
            class_filter = (gt_classes[:, 0] == expand_gt_classes[ex, 0])
            contain_filter = (box_utils.bbox_contain(expand_gt_boxes[ex:ex+1].astype(dtype=np.float32, copy=False), gt_boxes)[0] > contian_thres)
            cur_expand_filter = class_filter * contain_filter
            merge_filter += cur_expand_filter

        merge_boxes = gt_boxes[merge_filter]
        merge_classes = gt_classes[merge_filter]
        merge_scores = gt_scores[merge_filter]
        
        if merge:
            gt_boxes = gt_boxes[~merge_filter]
            gt_classes = gt_classes[~merge_filter]
            gt_scores = gt_scores[~merge_filter]
    
    gt_boxes = np.vstack((gt_boxes, expand_gt_boxes))
    gt_classes = np.vstack((gt_classes, expand_gt_classes))
    gt_scores = np.vstack((gt_scores, expand_gt_scores))


    if not fast:

        proposals = {'gt_boxes' : gt_boxes,
                    'gt_classes': gt_classes,
                    'gt_scores': gt_scores}
        return proposals

    else:
        proposals = {'gt_boxes' : gt_boxes,
                    'gt_classes': gt_classes,
                    'gt_scores': gt_scores,
                    'merge_boxes': merge_boxes,
                    'merge_classes': merge_classes,
                    'merge_scores': merge_scores,
                    'expand_gt_boxes': expand_gt_boxes,
                    'expand_gt_classes': expand_gt_classes,
                    'expand_gt_scores': expand_gt_scores}

        return proposals


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER,
        random_state=cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = box_utils.bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))

    return (overlaps > iou_threshold).astype(np.float32)


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, cfg.TRAIN.GRAPH_IOU_THRESHOLD)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)\
                [-1:(-1 - min(len(gt_scores_tmp), cfg.TRAIN.MAX_PC_NUM)):-1]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores,
                gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes,
                (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois,
                                           cfg.MODEL.BBOX_REG_WEIGHTS)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES + 1

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def get_proposal_clusters_semi(all_rois, proposals, im_labels):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    # need modifiied
    # if cfg.SUP.SEMI:
    cls_loss_weights[ig_inds] = 0.0
    # else:
    #     pass

    labels[bg_inds] = 0

    if cfg.MODEL.WITH_FRCNN:
        bbox_targets = _compute_targets(all_rois, gt_boxes[gt_assignment, :],
            labels)
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype) \
            * cls_loss_weights.reshape(-1, 1)
    else:
        bbox_targets, bbox_inside_weights, bbox_outside_weights = np.array([0]), np.array([0]), np.array([0])

    gt_assignment[bg_inds] = -1

    return labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights


def get_proposal_clusters(all_rois, proposals, im_labels):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    # need modifiied
    if cfg.SUP.SEMI:
        cls_loss_weights[ig_inds] = 0.0
    else:
        pass

    labels[bg_inds] = 0

    if cfg.MODEL.WITH_FRCNN:
        bbox_targets = _compute_targets(all_rois, gt_boxes[gt_assignment, :],
            labels)
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype) \
            * cls_loss_weights.reshape(-1, 1)
    else:
        bbox_targets, bbox_inside_weights, bbox_outside_weights = np.array([0]), np.array([0]), np.array([0])

    gt_assignment[bg_inds] = -1

    return labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights


class PCLLosses(nn.Module):

    def forward(ctx, pcl_probs, labels, cls_loss_weights, gt_assignments):
        cls_loss = 0.0
        weight = cls_loss_weights.view(-1).float()
        labels = labels.view(-1)
        gt_assignments = gt_assignments.view(-1)

        for gt_assignment in gt_assignments.unique():
            inds = torch.nonzero(gt_assignment == gt_assignments,
                as_tuple=False).view(-1)
            if gt_assignment == -1:
                assert labels[inds].sum() == 0
                cls_loss -= (torch.log(pcl_probs[inds, 0].clamp(1e-9, 10000))
                         * weight[inds]).sum()
            else:
                assert labels[inds].unique().size(0) == 1
                label_cur = labels[inds[0]]
                cls_loss -= torch.log(
                    pcl_probs[inds, label_cur].clamp(1e-9,  10000).mean()
                    ) * weight[inds].sum()

        return cls_loss / max(float(pcl_probs.size(0)), 1.)


class OICRLosses(nn.Module):
    def __init__(self):
        super(OICRLosses, self).__init__()

    def forward(self, prob, labels, cls_loss_weights, gt_assignments, eps = 1e-6, imlabel_=None):
        loss = torch.log(prob + eps)[range(prob.size(0)), labels]
        loss *= -cls_loss_weights
        ret = loss.mean()

        if cfg.SUP.ADD_NEG_LOSS:
            labels_NEG = imlabel_.clamp(0, 1)
            neg_loss = -(1 - labels_NEG) * torch.log(1 - prob[:, 1:] + eps)
            neg_num = (20 - labels_NEG.sum()) * prob.shape[0]
            mean_neg_loss = neg_loss.sum() / neg_num

            ret += mean_neg_loss * cfg.SUP.NEG_WEIGHT

        return ret

class BLOBLoss(nn.Module):
    def __init__(self):
        super(BLOBLoss, self).__init__()

    def forward(self, mil_result, refine_result, blob_conv, rois, labels, original_shape):

        # @jit(nopython=True)
        def Output_M(proposal_sorces, rois_, M):
            for temp_inds in range(rois_.shape[0]):
                r = rois_[temp_inds]
                target_field = M[int(r[1]):int(r[3]), int(r[0]):int(r[2])]
                M_scores = proposal_sorces[temp_inds]
                M_scores[np.where(M_scores < 0.3)] = 0
                add_score = np.zeros_like(target_field) + np.reshape(M_scores, [1, 1, -1])
                M[int(r[1]):int(r[3]), int(r[0]):int(r[2])] += add_score
            return M

        gpu2np = lambda a : a.detach().cpu().numpy()

        average_refine = sum(refine_i_ for refine_i_ in refine_result) / len(refine_result)
        
        if average_refine.shape[-1] != labels.shape[-1]:
            average_refine = average_refine[:, 1:]

        H, W = original_shape
        no_valid_class = torch.where(labels[0] != 1)[0]
        valid_class = torch.where(labels[0] == 1)[0]
        valid_class_num = valid_class.shape[0]

        M = np.zeros([H, W, valid_class_num], dtype=np.float32)
        valid_refine_scores = average_refine[:, valid_class]
        numpy_M = Output_M(gpu2np(valid_refine_scores), gpu2np(rois[:, 1:]), M)

        reconstruct_M = torch.from_numpy(numpy_M).to(average_refine.device)        

        reconstruct_M_min = reconstruct_M.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]

        reconstruct_M_max = reconstruct_M.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]

        reconstruct_M = (reconstruct_M - reconstruct_M_min) / (reconstruct_M_max - reconstruct_M_min + 1e-6)

        reconstruct_M = torch.nn.functional.interpolate(reconstruct_M.permute(2, 0, 1).unsqueeze(0), blob_conv.shape[-2:]).squeeze(0)

        if not cfg.MODEL.WITH_BLOB_CONV:
            blob_loss_ = self.loss_calculate_direct(reconstruct_M, blob_conv)
        else:
            # blob_loss_ = self.loss_calculate_sigmoid(reconstruct_M, blob_conv, valid_class)
            blob_loss_ = self.loss_calculate_class(reconstruct_M, blob_conv, valid_class, no_valid_class)

        return blob_loss_


    def loss_calculate_class(self, reconstruct_M, blob_conv, valid_class, no_valid_class):

        sigmoid_blob = blob_conv.clamp(min = 1e-6, max = 1 - 1e-6)

        max_x_blob = sigmoid_blob.max(-2)[0]
        max_y_blob = sigmoid_blob.max(-1)[0]

        max_x_weight = reconstruct_M.max(-2)[0].detach()
        max_y_weight = reconstruct_M.max(-1)[0].detach()

        max_x_label = torch.where(max_x_weight >= cfg.MODEL.BLOB_LOSS_THRES, 
                                torch.ones_like(max_x_weight, device=max_x_weight.device),
                                torch.zeros_like(max_x_weight, device=max_x_weight.device)).detach()
        
        max_y_label = torch.where(max_y_weight >= cfg.MODEL.BLOB_LOSS_THRES, 
                                torch.ones_like(max_y_weight, device=max_y_weight.device),
                                torch.zeros_like(max_y_weight, device=max_y_weight.device)).detach()
        
        # blob_loss_x_p = (-torch.log(max_x_blob[valid_class]) * max_x_label - torch.log(1 - max_x_blob[valid_class]) * (1 - max_x_label)).mean()

        blob_loss_x_p = (-torch.log(max_x_blob[valid_class]) * max_x_label).mean()

        blob_loss_x_n = (-torch.log(1 - max_x_blob[no_valid_class])).mean()

        blob_loss_y_p = (-torch.log(max_y_blob[valid_class]) * max_y_label).mean()

        blob_loss_y_n = (-torch.log(1 - max_y_blob[no_valid_class])).mean()

        blob_loss = blob_loss_x_p  + blob_loss_x_n + blob_loss_y_p + blob_loss_y_n

        return blob_loss


    
    def loss_calculate_sigmoid(self, reconstruct_M, blob_conv, valid_class):

        assert blob_conv.shape[0] == 1

        sigmoid_blob = blob_conv[0]

        max_x_blob = sigmoid_blob.max(0)[0].unsqueeze(0)
        max_y_blob = sigmoid_blob.max(1)[0].unsqueeze(0)

        max_x_weight = reconstruct_M.max(-2)[0].detach()
        max_y_weight = reconstruct_M.max(-1)[0].detach()

        mask_M = torch.where(reconstruct_M.max(0)[0] > cfg.MODEL.BLOB_LOSS_THRES,
                            torch.ones_like(sigmoid_blob, device=max_x_weight.device),
                            torch.zeros_like(sigmoid_blob, device=max_x_weight.device)).detach()

        max_x_label = torch.where(max_x_weight >= cfg.MODEL.BLOB_LOSS_THRES, 
                                torch.ones_like(max_x_weight, device=max_x_weight.device),
                                torch.zeros_like(max_x_weight, device=max_x_weight.device)).detach()
        
        max_y_label = torch.where(max_y_weight >= cfg.MODEL.BLOB_LOSS_THRES, 
                                torch.ones_like(max_y_weight, device=max_y_weight.device),
                                torch.zeros_like(max_y_weight, device=max_y_weight.device)).detach()
        
        blob_loss_x = (-torch.log(max_x_blob) * max_x_label * max_x_weight).sum(-1) / max_x_label.sum(-1)

        blob_loss_x = blob_loss_x.sum()

        blob_loss_y = (-torch.log(max_y_blob) * max_y_label * max_y_weight).sum(-1) / max_y_label.sum(-1)

        blob_loss_y = blob_loss_y.sum()

        blob_loss = blob_loss_x + blob_loss_y

        clamp_blob = sigmoid_blob.clamp(min = 1e-6, max = 1 - 1e-6)

        neg_loss = -((1 - mask_M) * torch.log(1 - clamp_blob)).sum() / ((1 - mask_M).sum() + 1e-6)

        blob_loss += neg_loss

        return blob_loss
    
    def loss_calculate_direct(self, reconstruct_M, blob_conv):
        assert blob_conv.shape[0] == 1

        sigmoid_blob = torch.sigmoid(blob_conv.squeeze(0).mean(dim=0))

        max_x_blob = sigmoid_blob.max(0)[0].unsqueeze(0)
        max_y_blob = sigmoid_blob.max(1)[0].unsqueeze(0)

        max_x_weight = reconstruct_M.max(-2)[0].detach()
        max_y_weight = reconstruct_M.max(-1)[0].detach()

        max_x_label = torch.where(max_x_weight >= cfg.MODEL.BLOB_LOSS_THRES, 
                                torch.ones_like(max_x_weight, device=max_x_weight.device),
                                torch.zeros_like(max_x_weight, device=max_x_weight.device)).detach()
        
        max_y_label = torch.where(max_y_weight >= cfg.MODEL.BLOB_LOSS_THRES, 
                                torch.ones_like(max_y_weight, device=max_y_weight.device),
                                torch.zeros_like(max_y_weight, device=max_y_weight.device)).detach()
        
        blob_loss_x = (-torch.log(max_x_blob) * max_x_label * max_x_weight).sum(-1) / (max_x_label.sum(-1) + 1e-6)

        blob_loss_x = blob_loss_x.sum()

        blob_loss_y = (-torch.log(max_y_blob) * max_y_label * max_y_weight).sum(-1) / (max_y_label.sum(-1) + 1e-6)

        blob_loss_y = blob_loss_y.sum()

        blob_loss = blob_loss_x + blob_loss_y

        return blob_loss
        



        







        




