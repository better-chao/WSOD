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
from model.pcl.pcl import PCL, PCLLosses, OICR, OICRLosses, BLOBLoss, SLV
from ops import RoIPool, RoIAlign
import modeling.pcl_heads as pcl_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.vgg_weights_helper as vgg_utils
from modeling.non_local_block import NONLocalBlock2D
from modeling.acol import CDBlock
from torchvision.ops import roi_pool
from modeling.ADL import ADL_cls
from modeling.RelationNeck import RelationNeck
from modeling.GraphNeck import GraphNeck
import os
import cv2
# from modeling.sub_center_oicr import Sub_Center


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


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.step = 1
        self.cdblock_flag = cfg.CDBLOCK.SWITCH

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        if cfg.NONLOCAL.SWITCH:
            self.Box_nolocal_module = NONLocalBlock2D(
                self.Conv_Body.dim_out,
                sub_sample=cfg.NONLOCAL.SUB_SAMPLE,
                bn_layer=cfg.NONLOCAL.BN_LAYER
            )

        if cfg.ADL.SWITCH:
            self.ADL_classifier = ADL_cls()
        
        if cfg.CDBLOCK.SWITCH:
            self.cdblock = CDBlock()

        if cfg.RN.SWITCH in ["RN", "graph"]:
            if cfg.RN.TYPE == "RN":
                self.Box_Head = RelationNeck()
            if cfg.RN.TYPE == "graph":
                self.Box_Head = GraphNeck(7*7*512, 4096, 4096, 64)
        else:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale, cfg)

        self.Box_MIL_Outs = pcl_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = pcl_heads.refine_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        if cfg.SLV.SWITCH:

            paramters = np.load(cfg.SLV.param_path, allow_pickle=True).item()
            self.gamma_param_list = [paramters[key] for key in paramters]
            if cfg.SLV.degree_add:
                self.wh_thres_list = self.search_thres(cfg.SLV.warmup_prior)
            else:
                self.wh_thres_list = self.search_thres(cfg.SLV.SLV_prior_thres)

        self.Refine_Losses = [OICRLosses() for i in range(cfg.REFINE_TIMES)]

        if cfg.MODEL.WITH_FRCNN:
            self.FRCNN_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
            self.Cls_Loss = OICRLosses()

        self._init_modules()
    
    def search_thres(self, SLV_prior_thres):
        x = np.arange(0, 10, cfg.SLV.BAND_WIDTH)
        wh_thres_list = []
        for cur_cls, params in enumerate(self.gamma_param_list):
            pdf_curve = st.gamma.pdf(x, params[0], scale=params[2])
            sorted_pdf = pdf_curve[np.argsort(-pdf_curve)]
            cur_thres = 0

            for v in sorted_pdf.tolist():
                if (pdf_curve[np.where(pdf_curve >= v)[0]] * cfg.SLV.BAND_WIDTH).sum() >= SLV_prior_thres:
                    cur_thres = v
                    break
            wh_thres_list.append(cur_thres)

        return wh_thres_list


    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            if cfg.MODEL.CONV_BODY.split('.')[0] == 'vgg16':
                vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, rois, labels, cur_step=None, gt_boxes=None, gt_boxes_classes=None, history_label=None, image_name=None):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels, cur_step=cur_step, gt_boxes=gt_boxes, gt_boxes_classes=gt_boxes_classes, history_label=history_label)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels, cur_step=cur_step, gt_boxes=gt_boxes, gt_boxes_classes=gt_boxes_classes, history_label=history_label,image_name=image_name)



    def _forward(self, data, rois, labels, cur_step, gt_boxes, gt_boxes_classes, history_label, image_name):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        self.step += 1

        return_dict = {}  # A dict to collect return variables
        # 骨干网络 未经过roi pooling
        conv_list = self.Conv_Body(im_data)

        blob_conv, adl_conv = conv_list[0].contiguous(), conv_list[1]

        

        self.final_conv = blob_conv
        # 经过roi_pooling 
        blob_conv = roi_pool(blob_conv, rois, [7, 7], self.Conv_Body.spatial_scale) # np,c,h,w


        # blob_conv = self.roi_feature_transform(
        #     blob_conv, rois,
        #     method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        #     resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        #     spatial_scale=self.Conv_Body.spatial_scale,
        #     sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        # ) 
        if self.cdblock_flag:
            # # blob_conv = self.cdblock(blob_conv)
            # # hot map
            # if cfg.CDBLOCK.CONV_ATT:
            #     blob_att = blob_conv.mean(1, keepdim=True)
            # else:
            #     blob_att = self.cdblock(blob_conv)

            # if cfg.CDBLOCK.TYPE == 1:
            #     att_v = F.softmax(blob_att, dim=-2)
            #     att_h = F.softmax(blob_att, dim=-1)
            #     att_map = att_v * att_h
            # elif cfg.CDBLOCK.TYPE == 2:
            #     att_v = F.softmax(blob_att, dim=-2)
            #     att_h = F.softmax(blob_att, dim=-1)
            #     att_map = torch.where(att_v > att_h, att_v, att_h)
            # elif cfg.CDBLOCK.TYPE == 3:
            #     att_map = torch.exp(blob_att)
            #     kernel_size = 3
            #     att_map_pad = F.pad(att_map, (1, 1, 1, 1), mode="constant", value=0)
            #     att_map = att_map / F.avg_pool2d(att_map_pad, kernel_size=kernel_size, stride=1) * (kernel_size**2)
            # else:
            #     raise ValueError("wrong cdblock type")

            # # attention apply
            # blob_conv = blob_conv * (1 + att_map)

            # 注意力增强
            blob_conv = self.cdblock(blob_conv)

        if not self.training:
            return_dict['blob_conv'] = torch.sigmoid(blob_conv.mean(1)).cpu().numpy()
            # print(return_dict['blob_conv'].shape)

        # 得到共享输入层
        box_feat = self.Box_Head(blob_conv, rois)
        # 输入到多示例学习层
        mil_score = self.Box_MIL_Outs(box_feat)
        # OICR 层 3个分类层的输出
        refine_score = self.Box_Refine_Outs(box_feat)
        # 这个是一个检测头
        if cfg.MODEL.WITH_FRCNN:
            cls_score, bbox_pred = self.FRCNN_Outs(box_feat)

        device = box_feat.device

        # 自训练
        if self.training:
            return_dict['losses'] = {}

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            # 多示例头的损失
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            # 存放梯度回传的所有loss 一般返回变量，但是这里将变量放入字典，返回到一个字典保存
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            # adl loss ，探索过程用到的模块，没用
            if cfg.ADL.SWITCH:
                logits = self.ADL_classifier(adl_conv)
                adl_loss = nn.BCELoss()(torch.sigmoid(logits), labels)
                return_dict['losses']['adl_loss'] = 0.5 * adl_loss


            # refinement loss
            # OICR的loss
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            im_labels_cuda = labels.clone()
            boxes = boxes[:, 1:]

            # sub_center_oicr = OICR(boxes, sum(refine_score) / 3, im_labels, sum(refine_score) / 3)

            # 计算标签
            # refine_score是三个OICR分类头的输出
            for i_refine, refine in enumerate(refine_score):
                if i_refine == 0:
                    # pcl_output是一个字典，包含伪框的位置信息、伪框的类别、伪框的分数，分数作为权重后面用到。
                    # 标签优化的代码在OICR这个类中
                    pcl_output = OICR(boxes, mil_score, im_labels, refine)
                else:
                    pcl_output = OICR(boxes, refine_score[i_refine - 1],
                                      im_labels, refine)

                refine_loss = self.Refine_Losses[i_refine](
                    refine,
                    Variable(torch.from_numpy(pcl_output['labels'])).to(device),
                    Variable(torch.from_numpy(pcl_output['cls_loss_weights'])).to(device),
                    Variable(torch.from_numpy(pcl_output['gt_assignment'])).to(device),
                    imlabel_=im_labels_cuda)

                # trick
                if i_refine == 0:
                    refine_loss *= 3.0

                # 每一个OICR分支存下来，也是存到字典
                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()

            # 计算检测头
            if cfg.MODEL.WITH_FRCNN:
                # 尝试，不用管
                if cfg.SLV.SWITCH:
                    labels, cls_loss_weights, bbox_targets, bbox_inside_weights, \
                        bbox_outside_weights = fast_rcnn_heads.get_fast_rcnn_slv_targets(
                            rois[:, 1:], sum(refine_score[-3:]) / 3, im_labels_cuda, im_data.shape[-2:], cfg.SLV.SLV_M_THRES, self.wh_thres_list, cfg.TRAIN.FG_THRESH, self.gamma_param_list, self, cur_step
                            )

                else:
                    # 得到检测头输出 ，包含较多信息
                    labels, cls_loss_weights, bbox_targets, bbox_inside_weights, \
                        bbox_outside_weights, extra_dict = fast_rcnn_heads.get_fast_rcnn_targets(
                            boxes, refine_score, im_labels, average=False)

                output_dir = "/home/zenghao/pcl/Outputs/vgg16_voc2007_more/Dec15-15-00-19_compute10_step"

                # self.plot_merge_img(im_data, extra_dict, output_dir)
                
                # 计算检测头分类损失和回归损失
                cls_loss, bbox_loss = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred,
                    Variable(torch.from_numpy(labels)).to(device),
                    Variable(torch.from_numpy(cls_loss_weights)).to(device),
                    Variable(torch.from_numpy(bbox_targets)).to(device),
                    Variable(torch.from_numpy(bbox_inside_weights)).to(device),
                    Variable(torch.from_numpy(bbox_outside_weights)).to(device))
                
                return_dict['losses']['cls_loss'] = cls_loss
                return_dict['losses']['bbox_loss'] = bbox_loss

            # if cfg.SLV.SWITCH:
            #     if cfg.SLV.degree_add:
            #         reweight = min((cur_step + 1) / cfg.SLV.WARMUP_STEP[1], 1)
            #     else:
            #         cur_epoch = cur_step / cfg.SOLVER.MAX_ITER
            #         reweight = min(max(cur_epoch - 0.2, 0) * 2, 1)
            #     print(cls_loss, bbox_loss)
            #     return_dict['losses']['cls_loss'] = cls_loss * reweight
            #     return_dict['losses']['bbox_loss'] = bbox_loss * reweight
            # else:
            #     return_dict['losses']['cls_loss'] = cls_loss
            #     return_dict['losses']['bbox_loss'] = bbox_loss


            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score
            # return_dict['refine_score'] = [refine_score[-1],refine_score[-1],refine_score[-1]]
            if cfg.MODEL.WITH_FRCNN:
                return_dict['cls_score'] = cls_score
                return_dict['bbox_pred'] = bbox_pred

        return return_dict

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
        for merge_num in range(merge_dict["expand_gt_boxes"].shape[0]):
            im_numpy = draw_info(
                im_numpy, 
                merge_dict["expand_gt_boxes"][merge_num],
                str(round(merge_dict["expand_gt_scores"][merge_num, 0], 2)),
                (0,255,0),
                2
            )

        vis_img_path = os.path.join(vis_img_dir, str(self.step) + ".jpg")
        cv2.imwrite(vis_img_path, im_numpy)


    def gen_feat_for_acol(self, out, thres):
        abs_high = 0.99
        # abs_low = 0.7
        bs = out.shape[0]
        mean_feat = out.mean(dim=1, keepdim=True)
        normal_feat = torch.sigmoid(mean_feat)

        flat_feat = normal_feat.view(bs, -1)
        max_val, _ = torch.max(flat_feat, dim=1, keepdim=True)
        thr_val = (max_val * abs_high).view(bs, 1, 1, 1)

        high_mask = (normal_feat >= thr_val).float().detach()
        high_mask = F.max_pool2d(input=high_mask,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=3 // 2)

        # self.low_level_weight = (mean_feat < thr_val).float().sum() / (mean_feat > 0).float().sum()
        # abs_high_count = (1 - high_mask).sum()
        # abs_low_count = (normal_feat < abs_low).float().sum() + 1e-6
        # self.low_level_weight = (mean_feat < thr_val).float().sum() / (mean_feat > 0).float().sum() * (abs_high_count / abs_low_count)

        return (1 - high_mask).float()

    
    def pool_feat_grad(self, pool_feat):
        # 该部分计算的是该框的类别的激活区域，对应于不同的类别
        self.eval()
        # pooled_feat_tmp = pool_feat.clone().detach()
        pooled_feat_tmp = pool_feat.clone().detach().requires_grad_(True)
        

        num_rois =  pooled_feat_tmp.shape[-4]
        tail_feat, _ = self.Box_Head(pooled_feat_tmp, None)
        cls_logit = self.Box_Refine_Outs(tail_feat)[-1]

        roi_gt = cls_logit.argmax(dim=1)
        one_hot = torch.zeros_like(cls_logit, dtype=torch.float32).cuda()
        one_hot[torch.arange(num_rois), roi_gt] = 1.0

        eval_loss = torch.sum(one_hot * cls_logit)
        self.zero_grad()
        pooled_feat_tmp.retain_grad()
        eval_loss.backward()

        num_channel = pooled_feat_tmp.shape[1]
        feat_grad = pooled_feat_tmp.grad.clone().detach()
        grad_channel_mean = torch.mean(feat_grad.view(num_rois, num_channel, -1), dim=2)
        cam_all = torch.sum(pooled_feat_tmp * grad_channel_mean.view(num_rois, num_channel, 1, 1), 1)
        cam_all = cam_all.view(num_rois, 49)
        th_mask_value = torch.sort(cam_all, dim=1, descending=True)[0][:, 18]
        th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, 49)
        mask_all_cuda = torch.where(cam_all > th_mask_value, torch.zeros(cam_all.shape).cuda(), torch.ones(cam_all.shape).cuda())
        mask_all = mask_all_cuda.reshape(num_rois, 7, 7).view(num_rois, 1, 7, 7)

        self.zero_grad()
        self.train()

        return mask_all


    # def get_blob_loss_weight(self, step_for_per_epoch = 5011):
    #     decimal_epoch = self.step / step_for_per_epoch
    #     blob_loss_weight = min(1, decimal_epoch * 0.1)
    #     return blob_loss_weight

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.
        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPool(resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoIAlign':
            xform_out = RoIAlign(
                resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

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
