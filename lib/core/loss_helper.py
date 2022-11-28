from datasets.json_dataset import JsonDataset
from core.test import im_detect_all, im_detect_bbox
from utils.timer import Timer
from core.config import cfg
from collections import defaultdict
import cv2
import logging
import numpy as np
import datetime
import json
from core.test import box_results_with_nms_and_limit
from utils.boxes import bbox_iou_np
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
import torch
import random



def dynamic_func(epoch, thres_update_type, all_epoch, low_bound=0.3):
        if thres_update_type == "constant":
            return cfg.TRAIN.THRESHOLD
        elif thres_update_type == "log_dynamic":
            return (np.log(epoch + low_bound) - np.log(low_bound)) / (np.log(all_epoch + low_bound) - np.log(low_bound))
        else:
            raise ValueError("type error!")



logger_label_helper = logging.getLogger(__name__)

# 更新训练集权重
class Label_Helper(object):
    def __init__(self, roidb, ratio_list, ratio_index):

        # 包含已经反转的图像
        self.train_roidb = []
        self.predict_reweight = False
        self.loss_split = False
        self.split_once = False
        self.loss_record = False
        self.split_cnt = 0
        for _ in roidb:
            tmp_dict = {}
            for i, j in _.items():
                tmp_dict[i] = j
            self.train_roidb.append(tmp_dict)

        self.record_dict = self.init_dict()
        self.record_score_dict = self.init_dict()
        # if self.predict_reweight:
        #     self.test_roidb = JsonDataset('voc_2007_trainval_pgf_2').get_roidb(
        #         proposal_file="data/selective_search_data/voc_2007_trainval.pkl",
        #         proposal_limit=False
        #     )

        self.ratio_list = ratio_list
        self.ratio_index = ratio_index

        self.thres = [0.6] * 20
        self.iou_thres = 0.5
        self.top_k_image = int(len(self.train_roidb) * 0.6)

# 保存训练集 标签的loss  和 category
    def init_dict(self):
        all_dict = {}


        for t_db in self.train_roidb:
            image_name = t_db["image"]
            all_dict[image_name] = {}
            loss_tmp, cls_tmp = [], []
            for box_num in range(t_db["gt_boxes_classes"].shape[-1]):
                loss_tmp.append([])
                cls_tmp.append(t_db["gt_boxes_classes"][0][box_num])
            all_dict[image_name]['loss'] = loss_tmp
            all_dict[image_name]['cls'] = cls_tmp
        return all_dict

# 执行一次更新操作 box_loss 分数和损失
    def update_meta(self, image_name, box_loss):
        # print(image_name)
        # print(image_name, len(box_loss["loss"]), len(self.record_dict[image_name]['loss']))
        for box_th in range(len(self.record_dict[image_name]['loss'])):
            self.record_dict[image_name]['loss'][box_th].append(box_loss["loss"][box_th])
            self.record_score_dict[image_name]['loss'][box_th].append(box_loss["score"][box_th])


    def update_db(self, model, cur_epoch):

        # ## 遍历所有的框并且要得出所有框的序列
        # init_cls_box_score = [[] for _ in range(20)]
        # for img_key in self.record_dict:
        #     for box_th in range(len(self.record_dict[img_key]['loss'])):
        #         cls_cur = self.record_dict[img_key]['cls'][box_th]
        #         loss_list = self.record_dict[img_key]['loss'][box_th]
        #         if len(loss_list) == 0:
        #             loss_cur = 100
        #         else:
        #             loss_cur = sum(loss_list) / len(loss_list)
        #         self.record_dict[img_key]['loss'][box_th] = loss_cur
        #         init_cls_box_score[cls_cur].append(loss_cur)
        
        # ## 计算得到topk的阈值，每个类别下的
        # cls_thres = []
        # for i in range(20):
        #     thres_num = int(self.top_k[i] * len(init_cls_box_score[i]))
        #     init_cls_box_score[i].sort()
        #     cls_thres.append(init_cls_box_score[i][-thres_num])
        

        # ##得到阈值之后，根据all_dict和threshold对train_db进行更新
        # for i in range(len(self.train_roidb)):
        #     entry = self.train_roidb[i]
        #     find_pre_loss = self.record_dict[entry['image']]['loss']
        #     for j in range(len(find_pre_loss)):
        #         assert self.record_dict[entry['image']]['cls'][j] == entry["gt_boxes_classes"][0][j]
        #         if self.loss_record:
        #             entry["history_label"][j].append(float(self.record_dict[entry['image']]['loss'][j]))
        #         else:
        #             if self.loss_split:
        #                 if not self.split_once:
        #                     entry["history_label"][j].append(float(self.record_dict[entry['image']]['loss'][j] < cls_thres[int(entry["gt_boxes_classes"][0][j])]))
        #                 elif self.split_cnt == 0:
        #                     entry["history_label"][j].append(float(self.record_dict[entry['image']]['loss'][j] < cls_thres[int(entry["gt_boxes_classes"][0][j])]))
        #                 else:
        #                     entry["history_label"][j].append(float(entry["history_label"][j][-1] != 0))
        #             else:
        #                 entry["history_label"][j].append(1.0)


        if cfg.TRAIN.STATIC_LOSS:
            loss_val = [[] for _ in range(20)]
            for i in range(len(self.train_roidb)):
                entry = self.train_roidb[i]
                for _ in range(len(entry['history_label'])):
                    tmp_cls = entry["gt_boxes_classes"][0][_]
                    loss_val[tmp_cls].append(entry['history_label'][_][0])
        else:
            loss_val = [[] for _ in range(20)]
            for img_key in self.record_dict:
                for box_th in range(len(self.record_dict[img_key]['loss'])):
                    cls_cur = self.record_dict[img_key]['cls'][box_th]
                    loss_list = self.record_dict[img_key]['loss'][box_th]
                    if len(loss_list) == 0:
                        loss_cur = 100
                    else:
                        loss_cur = sum(loss_list) / len(loss_list)
                    self.record_dict[img_key]['loss'][box_th] = loss_cur
                    loss_val[cls_cur].append(loss_cur)

        for _ in range(20):
            loss_val[_].sort()

        max_epoch = 14.0
        

        if cfg.TRAIN.DYNAMIC_TYPE == "linear":
            ratio = float(cur_epoch) / max_epoch
        elif cfg.TRAIN.DYNAMIC_TYPE == "p_linear":
            if cur_epoch < 3:
                ratio = 0.2
            elif cur_epoch < 6:
                ratio = 0.4
            elif cur_epoch < 9:
                ratio = 0.6
            elif cur_epoch < 12:
                ratio = 0.8
            else:
                ratio = 1.0
        elif cfg.TRAIN.DYNAMIC_TYPE == "sigmoid":
            ratio = 1.0 / (1.0 + np.e ** (max_epoch/2 - cur_epoch))
        elif cfg.TRAIN.DYNAMIC_TYPE == "exp":
            temp = cfg.TRAIN.EXP_TEMP
            ratio = (np.e ** ((cur_epoch - max_epoch) / temp) - np.e ** (-max_epoch/temp)) / (1 - np.e ** (-max_epoch/temp))
        elif cfg.TRAIN.DYNAMIC_TYPE == "log":
            low_bound = 0.01
            ratio = (np.log(cur_epoch + low_bound) - np.log(low_bound)) / (np.log(max_epoch + low_bound) - np.log(low_bound))
        elif cfg.TRAIN.DYNAMIC_TYPE == "constant":
            ratio = 1.0
        else:
            raise ValueError("error type")

        ratio = min(max(ratio, 0.0), 1.0)
        
        index_num = []
        ratio_list = []
        for _ in range(20):
            # 从大往小
            if cfg.TRAIN.TOP_TO_DOWN:
                true_ratio = 1 - (ratio * (1 - self.thres[_]))
            # 从小到大
            else:
                true_ratio = self.thres[_] + (1 - self.thres[_]) * ratio
            ratio_list.append(true_ratio)
            index_num.append(min(int(true_ratio * len(loss_val[_])), len(loss_val[_])-1))
        # thres_for_img = loss_val[index_num]
        # print(ratio_list)
        # exit(0)


        for i in range(len(self.train_roidb)):
            entry = self.train_roidb[i]
            # find_pre_loss = [_[0] for _ in self.entry['history_label']]
            for j in range(len(entry["history_label"])):
                box_cls_cur_tmp = entry["gt_boxes_classes"][0][j]
                # print(box_cls_cur_tmp)
                if cfg.TRAIN.STATIC_LOSS:
                    entry["history_label"][j].append(float(entry['history_label'][j][0] <= loss_val[box_cls_cur_tmp][index_num[box_cls_cur_tmp]]))
                else:
                    entry["history_label"][j].append(float(self.record_dict[entry['image']]['loss'][j] <= loss_val[box_cls_cur_tmp][index_num[box_cls_cur_tmp]]))
            

        # if self.split_once: self.split_cnt += 1
                # entry["history_label"][j].append(float(self.record_dict[entry['image']]['loss'][j]))
                # entry["history_label"][j].append(1.0)

        ## loss reweight
        if self.predict_reweight:
            for i in range(len(self.train_roidb)):
                entry = self.train_roidb[i]
                find_pre_score = self.record_score_dict[entry['image']]['loss']
                for j in range(len(find_pre_score)):
                    assert self.record_score_dict[entry['image']]['cls'][j] == entry["gt_boxes_classes"][0][j]
                    score_list = self.record_score_dict[entry['image']]['loss'][j]
                    if len(score_list) == 0: continue
                    if cur_epoch < cfg.TRAIN.REWEIGHT_EPOCH: continue
                    multiply_score = sum(score_list) / len(score_list)
                    entry["history_label"][j][-1] *= multiply_score
                        


        ## 重新初始化下一轮的all_record
        self.record_dict = self.init_dict()
        self.record_score_dict = self.init_dict()

# 更新标签权重 
    def update_dataloader(self, model, cur_epoch):

        self.update_db(model, cur_epoch)

        batchSampler = BatchSampler(
            sampler=MinibatchSampler(self.ratio_list, self.ratio_index),
            batch_size=cfg.NUM_GPUS,
            drop_last=True
        )

        dataset = RoiDataLoader(
            self.train_roidb,
            cfg.MODEL.NUM_CLASSES,
            training=True
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batchSampler,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            worker_init_fn=seed_worker,
            collate_fn=collate_minibatch
        )
        return dataloader



    def save_info(self, save_path):
        save_dict = {}

        gt_roidb = JsonDataset('voc_2007_trainval').get_roidb(
            proposal_file="data/selective_search_data/voc_2007_trainval.pkl",
            proposal_limit=False
        )

        gt_dict = {}

        for i, entry in enumerate(gt_roidb):
            gt_dict[entry['image']] = [entry["gt_boxes"], entry["gt_boxes_classes"][0]]

        for i, entry in enumerate(self.train_roidb):
            if entry["image"] not in save_dict:
                save_dict[entry['image']] = {"boxes":[]}

                for j in range(entry['gt_boxes'].shape[0]):
                    box_info = {
                        "xyxy": None,
                        "history": None,
                        "weight": None,
                        "label": None,
                        "right": None
                    }
                    box_info['xyxy'] = [float(_) for _ in entry['gt_boxes'][j].tolist()]
                    box_info['history'] = [float(_) for _ in entry['history_label'][j]]
                    box_info['weight'] = float(sum(box_info['history']) / len(box_info['history']))
                    box_info['label'] = int(entry['gt_boxes_classes'][0, j])

                    ious_m = bbox_iou_np(entry['gt_boxes'][j:j+1], gt_dict[entry['image']][0])
                    if ious_m[0].max() >= 0.5 and entry["gt_boxes_classes"][0, j] == gt_dict[entry['image']][1][np.argmax(ious_m[0])]:
                        box_info['right'] = True
                    else:
                        box_info['right'] = False
                    save_dict[entry['image']]['boxes'].append(box_info)
        
        save_json_data = json.dumps(save_dict)
        with open(save_path, 'w') as w_f:
            w_f.write(save_json_data)
            w_f.close()


















    