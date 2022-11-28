# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""generator a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
from tqdm import tqdm
import xml.etree.ElementTree as ET
import codecs
import json

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all, im_SLV_heatmap
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
from utils.io import save_object
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def get_eval_functions():
    # Determine which parent or child function should handle inference
    # Generic case that handles all network types other than RPN-only nets
    # and RetinaNet
    child_func = test_net
    parent_func = generator_meta

    return parent_func, child_func

def generate_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        all_results = {}
        dataset_name, proposal_file = args.dataset, args.proposal_file
        output_dir = args.output_dir
        parent_func(
            args,
            dataset_name,
            proposal_file,
            output_dir,
        )

    result_getter()


def save_xml(img_id, height, width, xml_path, batch_pred_boxes, batch_pred_scores, batch_pred_labels, dataset):
    with codecs.open(xml_path, 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + img_id + ".jpg" + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + "3" + '</depth>\n')
        xml.write('\t</size>\n')
        for b in range(batch_pred_boxes.shape[0]):
            xmin, ymin, xmax, ymax = int(batch_pred_boxes[b][0]), int(batch_pred_boxes[b][1]), int(batch_pred_boxes[b][2]), int(batch_pred_boxes[b][3])
            class_name = dataset.classes[batch_pred_labels[b]]
            #
            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + class_name + '</name>\n')
            xml.write('\t\t<wsod_scores>' + str(batch_pred_scores[b]) + '</wsod_scores>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
        xml.write('</annotation>')


def generate_xml(final_boxes, roidb, output_dir, dataset):
    xml_dir = os.path.join(output_dir, "generate_label", dataset.name + "_xml")
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    confidence_threshold = 0.0
    for img_inds, img_info in enumerate(roidb):
        gt_list = np.where(img_info["gt_classes"][0] == 1)[0].tolist()
        batch_pred_boxes = []
        batch_pred_scores = []
        batch_pred_labels = []
        for temp_cls in range(1, cfg.MODEL.NUM_CLASSES + 1):
            if (temp_cls - 1) in gt_list:
                prediction = final_boxes[temp_cls][img_inds]
                keep = np.where(prediction[:, -1] > confidence_threshold)[0]
                if keep.shape[0] == 0: continue

                selected_boxes = prediction[keep][:, 0:4]
                selected_scores = prediction[keep][:, -1]
                batch_pred_boxes.append(selected_boxes)
                batch_pred_scores.append(selected_scores)
                batch_pred_labels.append(np.full(selected_boxes.shape[0], temp_cls - 1, dtype=np.int32))

        if len(batch_pred_boxes) > 0:
            batch_pred_boxes = np.concatenate(batch_pred_boxes, axis=0) #[N. 4]
            batch_pred_scores = np.concatenate(batch_pred_scores, axis=0) #[N]
            batch_pred_labels = np.concatenate(batch_pred_labels, axis=0) #[N]

            width, height = img_info['width'], img_info['height']
            img_id = os.path.split(img_info["image"])[-1].split(".")[0]
            xml_path = os.path.join(xml_dir, img_id + ".xml")
            save_xml(img_id, height, width, xml_path, batch_pred_boxes, batch_pred_scores, batch_pred_labels, dataset)
        else:
            batch_pred_boxes = np.zeros((0, 4)) #[N. 4]
            batch_pred_scores = None
            batch_pred_labels = None

            width, height = img_info['width'], img_info['height']
            img_id = os.path.split(img_info["image"])[-1].split(".")[0]
            xml_path = os.path.join(xml_dir, img_id + ".xml")
            save_xml(img_id, height, width, xml_path, batch_pred_boxes, batch_pred_scores, batch_pred_labels, dataset)
        


def generate_coco(xml_dir, json_file, categories, bnd_id=1):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = os.listdir(xml_dir)
    with open("/home/zenghao/pcl/data/VOC2012/annotations/voc_2012_trainval.json", 'r') as f:
        gt_json = json.load(f)

    image2id = gt_json["images"]

    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}

    for line in list_fp:
        line = line.strip()
        # print(" Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = root.find('filename').text
        # 找到image id
        image_id = None
        for i2i in image2id:
            if i2i["file_name"] == filename:
                image_id = i2i["id"]
                break
        assert image_id is not None
        # 取出图片名字
        size = get_and_check(root, 'size', 1)
        # 图片的基本信息
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        # 处理每个标注的检测框
        for obj in get(root, 'object'):
            # 取出检测框类别名称
            category = get_and_check(obj, 'name', 1).text
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    #mmcv.dump(json_dict, json_file)
    print(type(json_dict))
    json_data = json.dumps(json_dict)
    with  open(json_file, 'w') as w:
        w.write(json_data)


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def generator_meta(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    

    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()
    test_timer = Timer()
    test_timer.tic()
    all_boxes = test_net(
        args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
    )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))

    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    final_boxes = empty_results(num_classes, num_images)
    for i, entry in enumerate(roidb):
        boxes = all_boxes[entry['image']]
        _, _, cls_boxes_i = box_results_with_nms_and_limit(boxes['scores'],
                                                         boxes['boxes'])
        extend_results(i, final_boxes, cls_boxes_i)
    # 唯一需要做的就是在这里加上之前的逻辑
    generate_xml(final_boxes, roidb, output_dir, dataset)
    xml_dir = os.path.join(output_dir, "generate_label", dataset.name + "_xml")
    json_file = os.path.join(output_dir,'generate_label', dataset.name + ".json")
    categories = {dataset.classes[_]:_ for _ in range(len(dataset.classes))}
    generate_coco(xml_dir, json_file, categories, bnd_id=1)
    # results = task_evaluation.evaluate_all(
    #     dataset, final_boxes, output_dir, test_corloc
    # )


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes = {}
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes']
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None

        im = cv2.imread(entry['image'])
        cls_boxes_i = im_detect_all(model, im, box_proposals, timers)

        # for test
        # cls_boxes_i = {}
        # cls_boxes_i["scores"] = np.random.rand(20, 21)
        # cls_boxes_i["boxes"] = np.asarray([[2.0, 2.0, 100.0, 100.0]]).repeat(21, axis=1).repeat(20, axis=0)

        all_boxes[entry['image']] = cls_boxes_i

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, eta
                )
            )

    cfg_yaml = yaml.dump(cfg)
    if 'train' in dataset_name:
        if ind_range is not None:
            det_name = 'discovery_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'discovery.pkl'
    else:
        if ind_range is not None:
            det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
