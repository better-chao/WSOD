import torch
import torch.nn as nn
from torchvision.ops import roi_pool
import numpy as np

class RelationModule(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64, isDuplication = False):
        super(RelationModule, self).__init__()
        self.isDuplication=isDuplication
        self.dim_g = geo_feature_dim
        self.relation1 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation2 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation3 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation4 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation5 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation6 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation7 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation8 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation9 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation10 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation11 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation12 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation13 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation14 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation15 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        self.relation16 = RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim)
        
        
    def forward(self, input_data):
        if(self.isDuplication):
            f_a, embedding_f_a, position_embedding =input_data
        else:
            f_a, position_embedding = input_data
        concat = self.relation1(f_a, position_embedding)
        concat = torch.cat((concat, self.relation2(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation3(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation4(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation5(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation6(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation7(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation8(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation9(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation10(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation11(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation12(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation13(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation14(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation15(f_a, position_embedding)), -1)
        concat = torch.cat((concat, self.relation16(f_a, position_embedding)), -1)
        return concat+f_a
        
    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping
        for name, m_child in self.named_children():
            if list(m_child.parameters()):  # if module has any parameter
                child_map, child_orphan = m_child.detectron_weight_mapping()
                for key, value in child_map.items():
                    new_key = name + '.' + key
                    d_wmap[new_key] = value
        return d_wmap, []

class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f_a, position_embedding):
        N,_ = f_a.size()

        position_embedding = position_embedding.view(-1,self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N,1,self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 )
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N,N)
        w_a = scaled_dot.view(N,N)

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N,N,1)
        w_v = w_v.view(N,1,-1)

        output = w_mn*w_v

        output = torch.sum(output,-2)
        return output
    
    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'WG.weight': 'WG_w',
            'WG.bias': 'WG_b',
            'WK.weight': 'WK_w',
            'WK.bias': 'WK_b',
            'WQ.weight': 'WQ_w',
            'WQ.bias': 'WQ_b',
            'WV.weight': 'WV_w',
            'WV.bias': 'WV_b',
        }
        return detectron_weight_mapping, []

class RelationNeck(nn.Module):

    def __init__(self, in_channels = 512, fc_features = 1024, n_relations = 16):
        # n_class includes the background
        super(RelationNeck, self).__init__()
        
        self.n_relations=n_relations
        self.fully_connected1 = nn.Linear(7*7*in_channels, fc_features)
        self.relu1 = nn.ReLU(inplace=True)

        self.fully_connected2 = nn.Linear(fc_features, fc_features)
        self.relu2 = nn.ReLU(inplace=True)
        if(n_relations>0):
            self.dim_g = int(fc_features/n_relations)
            self.relation1 = RelationModule(appearance_feature_dim=fc_features,
                                        key_feature_dim = self.dim_g, geo_feature_dim = self.dim_g)

            self.relation2 = RelationModule(appearance_feature_dim=fc_features,
                                        key_feature_dim=self.dim_g, geo_feature_dim=self.dim_g)

        self.dim_out = fc_features

    def forward(self, x, rois):
        """Forward the chain.
        We assume that there are :math:`N` batches.
        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """
        xy_indices_and_rois = rois[:, 1:5]
        indices_and_rois = torch.autograd.Variable(xy_indices_and_rois.contiguous())
        if(self.n_relations>0):
            position_embedding = PositionalEmbedding(indices_and_rois, dim_g = self.dim_g)

        x = x.view(x.size(0), -1)

        fc1 = self.relu1(self.fully_connected1(x))
        relation1 = self.relation1([fc1, position_embedding])
        fc2 = self.relu2(self.fully_connected2(relation1))
        relation2 = self.relation2([fc2, position_embedding])
        return relation2

    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping   
        for name, m_child in self.named_children():
            if name in ["relation1", "relation2"] and list(m_child.parameters()):  # if module has any parameter
                child_map, child_orphan = m_child.detectron_weight_mapping()
                for key, value in child_map.items():
                    new_key = name + '.' + key
                    d_wmap[new_key] = value
        mapping_to_detectron = d_wmap
        mapping_to_detectron["fully_connected1.weight"] = "fully_connected1_w"
        mapping_to_detectron["fully_connected1.bias"] = "fully_connected1_b"
        mapping_to_detectron["fully_connected2.weight"] = "fully_connected2_w"
        mapping_to_detectron["fully_connected2.bias"] = "fully_connected2_b"
        return mapping_to_detectron, []



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
