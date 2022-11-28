import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            "weight": "w",
            "bias": "b"
        }
        return detectron_weight_mapping, []


class GraphNeck(nn.Module):
    def __init__(self, dim_in, laten_dim, dim_out, dim_g, bias = True):
        super(GraphNeck, self).__init__()
        self.adj_extractor = PositionalAdjExtract(dim_g)
        self.graph_layer1 = GraphConvolution(dim_in, laten_dim, bias)
        self.graph_layer2 = GraphConvolution(laten_dim, dim_out, bias)
        # self.droplayer1 = nn.Dropout()
        # self.droplayer2 = nn.Dropout()
        self.relu_layer1 = nn.ReLU(inplace=True)
        self.relu_layer2 = nn.ReLU(inplace=True)
        self.dim_g = dim_g
        self.dim_out = dim_out
    
    def forward(self, in_feature, rois):
        in_feature = in_feature.view(in_feature.shape[0], -1)
        adj = self.adj_extractor(rois[:, 1:5])
        out = self.graph_layer1(in_feature, adj)
        out = self.relu_layer1(out)
        # out = self.droplayer1(out)

        out = self.graph_layer2(out, adj)
        out = self.relu_layer2(out)
        # out = self.droplayer2(out)
        return out

    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping
        for name, m_child in self.named_children():
            if list(m_child.parameters()):  # if module has any parameter
                child_map, child_orphan = m_child.detectron_weight_mapping()
                for key, value in child_map.items():
                    new_key = name + '.' + key
                    d_wmap[new_key] = value
        return d_wmap, []

class SparseGraphNeck(nn.Module):
    def __init__(self, dim_in, laten_dim, dim_out, dim_g, bias = True):
        super(GraphNeck, self).__init__()
        self.adj_extractor = PositionalAdjExtract(dim_g)
        self.graph_layer1 = GraphConvolution(dim_in, laten_dim, bias)
        self.graph_layer2 = GraphConvolution(laten_dim, dim_out, bias)
        # self.droplayer1 = nn.Dropout()
        # self.droplayer2 = nn.Dropout()
        self.relu_layer1 = nn.ReLU(inplace=True)
        self.relu_layer2 = nn.ReLU(inplace=True)
        self.dim_g = dim_g
        self.dim_out = dim_out
    
    def forward(self, in_feature, rois, rois_mask):
        in_feature = in_feature.view(in_feature.shape[0], -1)
        adj = self.adj_extractor(rois[:, 1:5])
        out = self.graph_layer1(in_feature, adj)
        out = self.relu_layer1(out)
        # out = self.droplayer1(out)

        out = self.graph_layer2(out, adj)
        out = self.relu_layer2(out)
        # out = self.droplayer2(out)
        return out

    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping
        for name, m_child in self.named_children():
            if list(m_child.parameters()):  # if module has any parameter
                child_map, child_orphan = m_child.detectron_weight_mapping()
                for key, value in child_map.items():
                    new_key = name + '.' + key
                    d_wmap[new_key] = value
        return d_wmap, []

class PositionalAdjExtract(nn.Module):
    def __init__(self, dim_g = 64):
        super(PositionalAdjExtract, self).__init__()
        self.WG = nn.Linear(dim_g, 1, bias=True)
        self.dim_g = dim_g
        
    
    def forward(self, rois):
        Embedding = PositionalEmbedding(rois, dim_g=self.dim_g)
        fc_Embedding = self.WG(Embedding).squeeze(-1)
        adj = torch.nn.Softmax(dim=1)(torch.log(torch.clamp(fc_Embedding, min = 1e-6)))
        return adj

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'WG.weight': 'WG_w',
            'WG.bias': 'WG_b',
        }
        return detectron_weight_mapping, []

def PositionalEmbedding( f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    # print(embedding.shape)

    return embedding
