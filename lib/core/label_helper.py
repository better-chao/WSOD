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

class Label_Helper(object):
    def __init__(self, roidb, ratio_list, ratio_index):

        # 包含已经反转的图像
        self.train_roidb = []
        for _ in roidb:
            tmp_dict = {}
            for i, j in _.items():
                tmp_dict[i] = j
            self.train_roidb.append(tmp_dict)

        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.test_roidb = JsonDataset('voc_2007_trainval_pgf').get_roidb(
            proposal_file="data/selective_search_data/voc_2007_trainval.pkl",
            proposal_limit=False
        )
        self.iou_thres = 0.5
        self.all_epoch = cfg.SOLVER.MAX_ITER // len(self.train_roidb) + 1

    
    def update_db(self, model, cur_epoch):
        model.eval()

        timers = defaultdict(Timer)
        det_all = {}
        num_images = len(self.test_roidb)
        start_ind, end_ind = 0, num_images
        for i_th, entry in enumerate(self.test_roidb):
            im = cv2.imread(entry['image'])

            # det_i = im_detect_all(model, im, entry['boxes'], timers)
            # _, _, det_i_ = box_results_with_nms_and_limit(det_i["scores"], det_i['boxes'])
            
            scores, pred_boxes, _, _ = im_detect_bbox(model, im, 576, 2000, entry['boxes'])
            if not cfg.TRAIN.NORMAL:
                _, _, det_i_ = box_results_with_nms_and_limit(scores, pred_boxes)
                det_all[entry['image']] = det_i_[1:]
            else:
                det_all[entry['image']] = [pred_boxes[:, 4:], scores[:, 1:]]

            # det_all[entry['image']] = [pred_boxes[:, 4:], scores[:, 1:]]

            # test
            # det_i_ = [np.asarray([[1, 1, 100, 100, 0.9]], dtype=np.float32) for _ in range(21)]
            # det_all[entry['image']] = det_i_[1:]
            

            # if i % 100 == 0:  # Reduce log file size
            #     ave_total_time = np.sum([t.average_time for t in timers.values()])
            #     eta_seconds = ave_total_time * (num_images - i - 1)
            #     eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            #     det_time = (
            #         timers['im_detect_bbox'].average_time
            #     )
            #     logger_label_helper.info(
            #         (
            #             'label update: range [{:d}, {:d}] of {:d}: '
            #             '{:d}/{:d} {:.3f}s (eta: {})'
            #         ).format(
            #             start_ind + 1, end_ind, num_images, start_ind + i + 1,
            #             start_ind + num_images, det_time, eta
            #         )
            #     )
        
        # update history label
        record_history_dict = {}
        for i in range(len(self.train_roidb)):
            if self.train_roidb[i]['flipped'] == False:
                entry = self.train_roidb[i]
                det_i = det_all[entry['image']]
                entry_copy = {}
                for k,v in entry.items():
                    entry_copy[k] = v
                self.train_roidb[i] = self.update_history(det_i, entry_copy, cur_epoch)

        # normalize
        if cfg.TRAIN.NORMAL:
            temperature = 10
            max_min_arr = [[0, 0] for _ in range(20)]
            for i in range(len(self.train_roidb)):
                if self.train_roidb[i]['flipped'] == False:
                    for box_th in range(len(self.train_roidb[i]["history_label"])):
                        box_th_cls = self.train_roidb[i]["gt_boxes_classes"][0][box_th]
                        box_th_score = self.train_roidb[i]["history_label"][box_th][-1]
                        max_min_arr[box_th_cls][0] += box_th_score
                        max_min_arr[box_th_cls][1] += 1
            for i in range(len(self.train_roidb)):
                if self.train_roidb[i]['flipped'] == False:
                    for box_th in range(len(self.train_roidb[i]["history_label"])):
                        box_th_cls = self.train_roidb[i]["gt_boxes_classes"][0][box_th]
                        average_score_for = max_min_arr[box_th_cls][0] / max_min_arr[box_th_cls][1]
                        cur_box_score_for = self.train_roidb[i]["history_label"][box_th][-1]
                        cur_box_score_for = 1 / (1 + np.e**(-(cur_box_score_for - average_score_for) * temperature))
                        self.train_roidb[i]["history_label"][box_th][-1] = cur_box_score_for

        # record for flipped
        for i in range(len(self.train_roidb)):
            if self.train_roidb[i]['flipped'] == False:
                record_history_dict[self.train_roidb[i]["image"]] = []
                for _ in self.train_roidb[i]["history_label"]:
                    tmp_list = []
                    for __ in _:
                        tmp_list.append(__)
                    record_history_dict[self.train_roidb[i]["image"]].append(tmp_list)

        # update for flipped
        for i in range(len(self.train_roidb)):
            if self.train_roidb[i]['flipped']:
                self.train_roidb[i]['history_label'] = record_history_dict[self.train_roidb[i]["image"]]

        if cfg.TRAIN.EXPAND_GT and cur_epoch >= cfg.TRAIN.REWEIGHT_EPOCH and (cur_epoch - cfg.TRAIN.REWEIGHT_EPOCH) % cfg.TRAIN.UPDATE_PER_TIME == 0:
             for i in range(len(self.train_roidb)):
                entry = self.train_roidb[i]
                det_i = det_all[entry['image']]

                self.expand_gt(det_i, entry, cur_epoch)


        model.train()

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
    

    


    def expand_gt(self, det, entry, cur_epoch):
        cur_gt_boxes, cur_gt_classes = entry['gt_boxes'], entry['gt_boxes_classes'][0]
        gt_weight = np.asarray([sum(h[1:]) / len(h[1:]) for h in entry["history_label"]])
        unique_classes = np.unique(cur_gt_classes)

        expand_gt_boxes = np.zeros((0, 4), dtype=np.float32)
        expand_gt_classes = np.zeros((1, 0), dtype=np.int32)
        expand_gt_history = []

        for u_c in unique_classes:
            prediction = det[u_c]
            if prediction.shape[0] == 0: continue


            arr = np.zeros_like(gt_weight, dtype=np.int32)

            # filter_key = ((gt_weight >= (arr + cfg.TRAIN.THRESHOLD)) * (cur_gt_classes == (arr + u_c)))


            filter_key = ((cur_gt_classes == (arr + u_c)))

            now_box = cur_gt_boxes[np.where(filter_key == True)[0], :].copy()


            if entry["flipped"] and now_box.shape[0] != 0:
                width = entry['width']
                oldx1 = now_box[:, 0].copy()
                oldx2 = now_box[:, 2].copy()
                now_box[:, 0] = width - oldx2 - 1
                now_box[:, 2] = width - oldx1 - 1


            for p in range(prediction.shape[0]):
                if now_box.shape[0] != 0:
                    ious_m = bbox_iou_np(prediction[p:p+1, 0:4], now_box)[0]
                    if ious_m.max() < 0.3 and prediction[p, -1] > 0.9:
                        insert_p = prediction[p:p+1, 0:4].copy()
                        if entry["flipped"]:
                            width = entry['width']
                            oldx1 = insert_p[:, 0].copy()
                            oldx2 = insert_p[:, 2].copy()
                            insert_p[:, 0] = width - oldx2 - 1
                            insert_p[:, 2] = width - oldx1 - 1
                        
                        expand_gt_boxes = np.concatenate([expand_gt_boxes, insert_p], axis=0)
                        expand_gt_classes = np.concatenate([expand_gt_classes, np.zeros((1, 1), dtype=np.int32) + int(u_c)], axis=1)
                        expand_gt_history.append([1, prediction[p, -1]])
                else:
                    insert_p = prediction[p:p+1, 0:4].copy()
                    if entry["flipped"]:
                        width = entry['width']
                        oldx1 = insert_p[:, 0].copy()
                        oldx2 = insert_p[:, 2].copy()
                        insert_p[:, 0] = width - oldx2 - 1
                        insert_p[:, 2] = width - oldx1 - 1
                        
                    expand_gt_boxes = np.concatenate([expand_gt_boxes, insert_p], axis=0)
                    expand_gt_classes = np.concatenate([expand_gt_classes, np.zeros((1, 1), dtype=np.int32) + int(u_c)], axis=1)
                    expand_gt_history.append([1, prediction[p, -1]])

        
        if expand_gt_boxes.shape[0] != 0:
            entry['gt_boxes'] = np.concatenate([entry['gt_boxes'], expand_gt_boxes], axis=0)
            entry['gt_boxes_classes'] = np.concatenate([entry['gt_boxes_classes'], expand_gt_classes], axis=1)
            entry['history_label'].extend(expand_gt_history)


    # def update_history(self, det, entry, cur_epoch):
        

    def update_history(self, det, entry, cur_epoch):
        if cfg.TRAIN.NORMAL:
            cur_gt_boxes, cur_gt_classes = entry['gt_boxes'], entry['gt_boxes_classes'][0]
            for i in range(cur_gt_boxes.shape[0]):
                prediction = det[0][:, int(4 * cur_gt_classes[i]):int(4 * cur_gt_classes[i] + 4)]
                if prediction.shape[0] == 0:
                    entry['history_label'][i].append(0.0)
                    continue
                ious_m = bbox_iou_np(cur_gt_boxes[i:i+1], prediction[:, 0:4])[0]
                if ious_m.max() >= self.iou_thres:
                    positive_box_index = np.where(ious_m >= self.iou_thres)[0]
                    score = det[1][:, int(cur_gt_classes[i])][positive_box_index].max()
                    entry['history_label'][i].append(float(score))
                else:
                    entry['history_label'][i].append(0.0)
            return entry
        else:
            cur_gt_boxes, cur_gt_classes = entry['gt_boxes'], entry['gt_boxes_classes'][0]
            for i in range(cur_gt_boxes.shape[0]):
                prediction = det[cur_gt_classes[i]]
                if prediction.shape[0] == 0:
                    entry['history_label'][i].append(0.0)
                    continue
                ious_m = bbox_iou_np(cur_gt_boxes[i:i+1], prediction[:, 0:4])[0]
                if ious_m.max() >= self.iou_thres:
                    score = prediction[np.argmax(ious_m), -1]
                    # if score < self.dynamic_func(cur_epoch):
                    #     entry['history_label'][i].append(0.0)
                    # else:
                    #     entry['history_label'][i].append(float(score))
                    entry['history_label'][i].append(float(score))
                else:
                    entry['history_label'][i].append(0.0)
            return entry


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


















    