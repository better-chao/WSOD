import torch.nn as nn
import torch
import torch.nn.functional as F
from core.config import cfg

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        detectron_weight_mapping["fc.0.weight"] = "fc_0_weight"
        detectron_weight_mapping["fc.0.bias"] = "fc_0_bias"
        detectron_weight_mapping["fc.2.weight"] = "fc_2_weight"
        detectron_weight_mapping["fc.2.bias"] = "fc_2_bias"
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.fc = nn.Linear(7 * 7 * 2, 1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        

    def forward(self, x):
        p_n = x.shape[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_c = torch.cat([avg_out, max_out], dim=1)

        p_x = self.sigmoid(self.fc(x_c.view(p_n, -1))).view(p_n, 1, 1, 1)
        x_ = self.sigmoid(self.conv1(x_c)) * p_x
        return x_

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        detectron_weight_mapping["fc.weight"] = "fc_weight"
        detectron_weight_mapping["fc.bias"] = "fc_bias"
        detectron_weight_mapping["conv1.weight"] = "conv1_weight"
        detectron_weight_mapping["conv1.bias"] = "conv1_bias"
        orphan_in_detectron = []
 
        return detectron_weight_mapping, orphan_in_detectron


# 注意力机制类
class CDBlock(nn.Module):
    def __init__(self, inplanes=512):
        super(CDBlock, self).__init__()
        if cfg.CDBLOCK.CHANNEL_ATT:
            self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)
        self.block_size = 3


    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask

        return block_mask
    

    def get_random_index(self, rois_num, device):
        p = torch.rand([rois_num]).cuda() + 0.2
        return torch.floor(p)[:, None, None, None]


    def get_random_block_mask(self, x, spatial_map, gamma=0.3):
        mask = (torch.rand(x.shape[0], 1, *x.shape[2:]) < (gamma / 9)).float()
        # mask = mask.to(x.device)
        return self._compute_block_mask(mask)


    def adl_drop_mask(self, attention, drop_thres=0.3):
        b_size = attention.size(0)
        thres_val_num = int(drop_thres * b_size * 7 * 7 / 9)
        flatten_att = attention.view(-1, b_size * 7 * 7)
        thres_val = torch.sort(flatten_att, dim=-1)[0][0, -thres_val_num]
        thr_val = thres_val.view(1, 1, 1, 1)
        return (attention > thr_val).float()

    def Dropblock_forward(self, x, spatial_map, channel_map):
        # spatial_mask = self.get_random_block_mask(x.detach(), None).to(x.device)
        spatial_mask = self.adl_drop_mask(spatial_map).to(x.device)
        mask = spatial_mask
        return mask


    def CBAM_forward(self, x):
        if cfg.CDBLOCK.CHANNEL_ATT:
            ca_map = self.ca(x)
        # x = ca_map * x
        sa_map = self.sa(x)
        if cfg.CDBLOCK.CHANNEL_ATT:
            x = sa_map * x * ca_map
        else:
            x = sa_map * x
        # return x, sa_map.detach(), ca_map.detach()
        return x, None, sa_map


    def forward(self, x):
        if not self.training:
            residual, ca_map, sa_map = self.CBAM_forward(x.clone())
            return self.relu(residual + x)
        else:
            residual, ca_map, sa_map = self.CBAM_forward(x.clone())
            # key_for_select = self.get_random_index(1, x.device).detach().item()
            # dropblock_mask = self.Dropblock_forward(x, sa_map, ca_map).detach()
            # output = (1 - key_for_select) * (self.relu(residual * x + x)) + key_for_select * (self.relu((1 - residual) * x + x))
            output = self.relu(residual + x)
            # output = self.relu(dropblock_mask * x)
            return output
    

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for name, m_child in self.named_children():
            if list(m_child.parameters()):
                child_map, child_orphan = m_child.detectron_weight_mapping()
                orphan_in_detectron.extend(child_orphan)
                for key, value in child_map.items():
                        new_key = name + '.' + key
                        detectron_weight_mapping[new_key] = value
        return detectron_weight_mapping, orphan_in_detectron


# class CDBlock(nn.Module):
#     def __init__(self):
#         super(CDBlock, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, 1, padding=0, bias=False)

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x

#     def detectron_weight_mapping(self):
#         detectron_weight_mapping = {}
#         detectron_weight_mapping["conv1.weight"] = "conv1_weight"
#         detectron_weight_mapping["conv1.bias"] = "conv1_bias"
#         orphan_in_detectron = []
#         return detectron_weight_mapping, orphan_in_detectron




