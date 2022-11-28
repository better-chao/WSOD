from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy.stats as st

from core.config import cfg
from ops import RoIPool, RoIAlign
import utils.blob as blob_utils
import utils.net as net_utils
import utils.vgg_weights_helper as vgg_utils
import modeling.fast_rcnn_heads as fast_rcnn_heads
from torchvision.ops import roi_pool
from modeling.ADL import ADL_cls
import cv2
import os


logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class fast_rcnn_builder(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.step = 1
        self.student_switch = False

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.Conv_Body.dim_out, None, self.Conv_Body.spatial_scale, cfg)
        
        self.FRCNN_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        if self.student_switch:
            self.student_1 = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
            self.student_2 = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
            self.student_3 = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)


        self._init_modules()


    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            if cfg.MODEL.CONV_BODY.split('.')[0] == 'vgg16':
                vgg_utils.load_pretrained_imagenet_weights(self)


    def forward(self, data, rois, labels, cur_step=None, gt_boxes=None, gt_boxes_classes=None, history_label=None,image_name=None):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels, cur_step=cur_step, gt_boxes=gt_boxes, gt_boxes_classes=gt_boxes_classes, history_label=history_label)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels, cur_step=cur_step, gt_boxes=gt_boxes, gt_boxes_classes=gt_boxes_classes, history_label=history_label, image_name=image_name)


    def _forward(self, data, rois, labels, cur_step, gt_boxes, gt_boxes_classes, history_label, image_name):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        return_dict = {}  # A dict to collect return variables

        imlabels_clone = labels.clone()

        conv_list = self.Conv_Body(im_data)

        blob_conv, adl_conv = conv_list[0].contiguous(), conv_list[1]

        

        blob_conv = roi_pool(blob_conv, rois, [7, 7], self.Conv_Body.spatial_scale)

        if not self.training:
            return_dict['blob_conv'] = torch.sigmoid(blob_conv.mean(1)).cpu().numpy()

        box_feat = self.Box_Head(blob_conv, rois)

        
        

        cls_score, bbox_pred = self.FRCNN_Outs(box_feat)

        if self.student_switch:
            student_cls_score_1, student_bbox_pred_1 = self.student_1(box_feat)
            student_cls_score_2, student_bbox_pred_2 = self.student_2(box_feat)
            student_cls_score_3, student_bbox_pred_3 = self.student_3(box_feat)

        if self.training:
            #TO DO
            self.step += 1
            device = box_feat.device
            return_dict["losses"] = {}
            boxes = rois.data.cpu().numpy()[:, 1:]
            im_labels = labels.data.cpu().numpy()
            # 计算弱监督标签的标签和损失值
            labels, cls_loss_weights, bbox_targets, bbox_inside_weights, \
                        bbox_outside_weights, gt_assignment = fast_rcnn_heads.get_fast_rcnn_teacher_targets(
                            boxes, gt_boxes, gt_boxes_classes, im_labels, history_label, cur_step)

            cls_loss, bbox_loss, loss_per_anno = fast_rcnn_heads.fast_rcnn_losses_2(
                    cls_score, bbox_pred,
                    Variable(torch.from_numpy(labels)).to(device),
                    Variable(torch.from_numpy(cls_loss_weights)).to(device),
                    Variable(torch.from_numpy(bbox_targets)).to(device),
                    Variable(torch.from_numpy(bbox_inside_weights)).to(device),
                    Variable(torch.from_numpy(bbox_outside_weights)).to(device),
                    imlabels_clone,
                    Variable(torch.from_numpy(gt_assignment)).to(device),
                    gt_boxes.shape[-2])

            output_dir = "/home/zenghao/pcl/Outputs/vgg16_voc2007_more/Dec15-15-00-19_compute10_step"

            # if self.step % 60 == 0:
            #     self.plot_merge_img(im_data, student_proposals, output_dir)

            if self.student_switch:

                student_labels_1, student_cls_loss_weights_1, student_bbox_targets_1, student_bbox_inside_weights_1, \
                        student_bbox_outside_weights_1, student_gt_assignment_1, student_proposals_1 = fast_rcnn_heads.get_fast_rcnn_student_targets(
                            boxes, gt_boxes, gt_boxes_classes, im_labels, history_label, cur_step, F.softmax(cls_score, -1))
                
                student_cls_loss_1, student_bbox_loss_1 = fast_rcnn_heads.fast_rcnn_losses(
                        student_cls_score_1, student_bbox_pred_1,
                        Variable(torch.from_numpy(student_labels)).to(device),
                        Variable(torch.from_numpy(student_cls_loss_weights)).to(device),
                        Variable(torch.from_numpy(student_bbox_targets)).to(device),
                        Variable(torch.from_numpy(student_bbox_inside_weights)).to(device),
                        Variable(torch.from_numpy(student_bbox_outside_weights)).to(device),
                        imlabels_clone)
                 
                # if cur_step < 12 and cfg.TRAIN.LABEL_UPDATE else 1.0
                return_dict['losses']['student_cls_loss_1'] = student_cls_loss_1
                return_dict['losses']['student_bbox_loss_1'] = student_bbox_loss_1
            
            return_dict['losses']['cls_loss'] = cls_loss
            return_dict['losses']['bbox_loss'] = bbox_loss

            
            return_dict['record_anno_loss'] = loss_per_anno

            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
        else:
            return_dict["rois"] = rois
            # bbox_test_scores = student_cls_score
            # bbox_test_pred = student_bbox_pred
            bbox_test_scores = cls_score
            bbox_test_pred = bbox_pred
            return_dict['refine_score'] = [bbox_test_scores, bbox_test_scores, bbox_test_scores]
            return_dict['cls_score'] = bbox_test_scores
            return_dict['bbox_pred'] = bbox_test_pred
        return return_dict

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    def plot_merge_img(self, im_data, merge_dict, output_dir):
        vis_img_dir = os.path.join(output_dir, "merge_vis")
        if not os.path.exists(vis_img_dir):
            os.makedirs(vis_img_dir)
        merge_boxes_name = "gt_boxes"
        merge_scores_name = "gt_scores"
        merge_classes_name = "gt_classes"
        if merge_boxes_name not in merge_dict or merge_dict[merge_boxes_name] is None: return
        
        def draw_info(image, box, text_info, color=(0,0,255), thickness=1):
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
            cv2.putText(image, text_info, (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            return image

        im_numpy = (np.transpose(im_data[0].detach().cpu().numpy(), [1, 2, 0])).astype(np.uint8).copy()
        # print("plot")

        # plot merge boxes
        for merge_num in range(merge_dict[merge_boxes_name].shape[0]):
            im_numpy = draw_info(
                im_numpy, 
                merge_dict[merge_boxes_name][merge_num],
                str(merge_dict[merge_classes_name][merge_num])+ " " + str(round(merge_dict[merge_scores_name][merge_num, 0], 2)),
                (0,0,255),
                2
            )

        # plot expand boxes
        # for merge_num in range(merge_dict["expand_gt_boxes"].shape[0]):
        #     im_numpy = draw_info(
        #         im_numpy, 
        #         merge_dict["expand_gt_boxes"][merge_num],
        #         str(round(merge_dict["expand_gt_scores"][merge_num, 0], 2)),
        #         (0,255,0),
        #         2
        #     )

        vis_img_path = os.path.join(vis_img_dir, str(self.step) + ".jpg")
        cv2.imwrite(vis_img_path, im_numpy)

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
