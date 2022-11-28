import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils

from modeling.dropblock import DropBlock


class mil_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mil_score0 = nn.Linear(dim_in, dim_out)
        self.mil_score1 = nn.Linear(dim_in, dim_out)

        if cfg.SPLITE.SWITCH:
            self.detect_head = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True), 
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True), 
                nn.Dropout(),
                nn.Linear(4096, dim_out)
            )

            self.spatial_attention_conv = nn.Conv2d(2, 1, [3, 1], padding=[1, 0])

            self.split_max_pooling = nn.MaxPool1d(7, 1)
            self.split_avg_pooling = nn.AvgPool1d(7, 1)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.mil_score0.weight, std=0.01)
        init.constant_(self.mil_score0.bias, 0)
        init.normal_(self.mil_score1.weight, std=0.01)
        init.constant_(self.mil_score1.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'mil_score0.weight': 'mil_score0_w',
            'mil_score0.bias': 'mil_score0_b',
            'mil_score1.weight': 'mil_score1_w',
            'mil_score1.bias': 'mil_score1_b'
        }
        if cfg.SPLITE.SWITCH:
            detectron_weight_mapping["detect_head.0.weight"] = "detect_head_0_weight"
            detectron_weight_mapping["detect_head.0.bias"] = "detect_head_0_bias"
            detectron_weight_mapping["detect_head.3.weight"] = "detect_head_3_weight"
            detectron_weight_mapping["detect_head.3.bias"] = "detect_head_3_bias"
            detectron_weight_mapping["detect_head.6.weight"] = "detect_head_6_weight"
            detectron_weight_mapping["detect_head.6.bias"] = "detect_head_6_bias"

            detectron_weight_mapping["spatial_attention_conv.weight"] = "spatial_attention_conv_weight"
            detectron_weight_mapping["spatial_attention_conv.bias"] = "spatial_attention_conv_bias"

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    
    def spatial_attention_forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.spatial_attention_conv(x)
        return torch.sigmoid(x)


    def split_block(self, roi_feature_clone):
        if self.training:
            sigmoid_activate = torch.sigmoid(roi_feature_clone.mean(dim=1, keepdim=True))

            attention_weight = self.generate_weight(roi_feature_clone.device)

            sigmoid_activate = sigmoid_activate * attention_weight

            out_bg = 1 - sigmoid_activate

            out = torch.cat([sigmoid_activate, out_bg], dim=1)

            scores = self.gumbel_softmax(out, dim=1, tau=0.01, hard=True, eps=1e-10)

            mask = scores[:, 0]

            detect_feature = roi_feature_clone * mask[:, None, :, :]
        else:
            detect_feature = roi_feature_clone

        return detect_feature
    
    def sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)


    def generate_weight(self, device, declay_ratio=0.1):
        init_weights = torch.zeros([1, 1, 7, 7], device=device)
        for i in range(4):
            init_weights[0, 0, i:7-i, i:7-i] = 1 - 0.1 * i
        return init_weights


    def gumbel_softmax(self, logits, tau=1, hard=False, eps=1e-10, dim=-1):
        # type: (Tensor, float, bool, float, int) -> Tensor
        device = logits.device
        gumbels = self.sample_gumbel(logits.shape, device, eps)
        gumbels = (logits.add(1e-10).log() + gumbels) / tau  # ~Gumbel(logits,tau)
        # y_soft = gumbels.softmax(dim)
        y_soft = torch.exp(F.log_softmax(gumbels, dim))

        if hard:
            # Reparametrization trick.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Straight through.
            ret = y_soft

        return ret

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        # print(roi_feature_clone.shape)
        mil_score1 = self.mil_score1(x)
        mil_score0 = self.mil_score0(x)
        mil_score = F.softmax(mil_score0, dim=0) * F.softmax(mil_score1, dim=1)

        return mil_score

class acol_head(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.acol_neck = nn.Sequential(
            nn.Linear(dim_in * 7 * 7, 4096),
            nn.ReLU(inplace=True), 
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), 
            # nn.Dropout(),
            # nn.Linear(4096, self.num_classes)
        )

        self.acol_cls_head = nn.Linear(4096, dim_out)
        self.acol_det_head = nn.Linear(4096, dim_out)
        

    def forward(self, x):
        latent_feat = self.acol_neck(x)
        acol_cls_grade = F.softmax(self.acol_cls_head(latent_feat), dim=1)
        acol_det_grade = F.softmax(self.acol_det_head(latent_feat), dim=0)
        return acol_det_grade * acol_cls_grade

    
    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        
        detectron_weight_mapping["acol_neck.0.weight"] = "acol_neck_0_weight"
        detectron_weight_mapping["acol_neck.0.bias"] = "acol_neck_0_bias"
        detectron_weight_mapping["acol_neck.2.weight"] = "acol_neck_2_weight"
        detectron_weight_mapping["acol_neck.2.bias"] = "acol_neck_2_bias"
        detectron_weight_mapping["acol_cls_head.weight"] = "acol_cls_head_weight"
        detectron_weight_mapping["acol_cls_head.bias"] = "acol_cls_head_bias"
        detectron_weight_mapping["acol_det_head.weight"] = "acol_det_head_weight"
        detectron_weight_mapping["acol_det_head.bias"] = "acol_det_head_bias"


        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    
class blob_conv_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.segment_score = nn.Conv2d(dim_in, dim_out, (1, 1))
        # self.blob_conv1 = nn.Conv2d(dim_in, dim_in, (3, 3), padding=1)
        # self.blob_bn_1 = nn.BatchNorm2d(dim_in)
        # self.blob_relu = nn.ReLU(inplace=True)
        # self.blob_conv2 = nn.Conv2d(dim_in, dim_out, (3, 3), padding=1)
        # self.blob_bn_2 = nn.BatchNorm2d(dim_out)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.segment_score.weight, std=0.01)
        init.constant_(self.segment_score.bias, 0)
        # init.normal_(self.blob_conv1.weight, std=0.01)
        # init.constant_(self.blob_conv1.bias, 0)
        # init.normal_(self.blob_conv2.weight, std=0.01)
        # init.constant_(self.blob_conv2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            "segment_score.weight": "segment_score.weight",
            "segment_score.bias": "segment_score.bias"
            # 'blob_conv1.weight': 'blob_conv1_w',
            # 'blob_conv1.bias': 'blob_conv1_b',
            # 'blob_bn_1.weight': 'blob_bn_1_w',
            # 'blob_bn_1.bias': 'blob_bn_1_b',
            # 'blob_conv2.weight': 'blob_conv2_w',
            # 'blob_conv2.bias': 'blob_conv2_b',
            # 'blob_bn_2.weight': 'blob_bn_2_w',
            # 'blob_bn_2.bias': 'blob_bn_2_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        out = self.segment_score(x)
        # out = self.blob_conv1(x)
        # out = self.blob_bn_1(out)
        # out = self.blob_relu(out)
        # out = self.blob_conv2(out)
        # out = self.blob_bn_2(out)
        return out.sigmoid()


class refine_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.refine_score = []
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)

        self._init_weights()

    def _init_weights(self):
        for i_refine in range(cfg.REFINE_TIMES):
            init.normal_(self.refine_score[i_refine].weight, std=0.01)
            init.constant_(self.refine_score[i_refine].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        for i_refine in range(cfg.REFINE_TIMES):
            detectron_weight_mapping.update({
                'refine_score.%d.weight' % i_refine: 'refine_score%d_w' % i_refine,
                'refine_score.%d.bias' % i_refine: 'refine_score%d_b' % i_refine
            })
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        refine_score = [F.softmax(refine(x), dim=1) for refine in self.refine_score]

        return refine_score


class slv_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.slv_cls = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.slv_cls.weight, std=0.01)
        init.constant_(self.slv_cls.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        
        detectron_weight_mapping["slv_cls.weight"] = "slv_cls_weight"
        detectron_weight_mapping["slv_cls.bias"] = "slv_cls_bias"
            
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        slv_score = F.softmax(self.slv_cls(x), dim=1)

        return slv_score


def mil_losses(cls_score, labels):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)

    return loss.mean()
