import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore")


class Sub_Center(nn.Module):
    def __init__(self, num_features, num_classes, bag_size = 3, beta=0.033, alpha=0.05):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        # self.knn = knn  # Knn for neighborhood invariance

        # Exemplar memory
        self.em = nn.Parameter(torch.randn(num_classes + 1, bag_size, num_features))

    def forward(self, inputs, targets, target_weights, epoch=None):

        if not self.training:
            # alpha = self.alpha  * epoch
            inputs = F.normalize(inputs)
            num_classes, bag_size, num_features = self.em.shape
            bs = inputs.shape[0]
            outputs = inputs.mm(self.em.view(-1, num_features).t())
            return outputs.view(bs, num_classes, bag_size).max(-1)[0]

        inputs = inputs
        centers = self.em

        num_classes, bag_size, num_features = centers.shape
        bs = inputs.shape[0]
        inputs = inputs.mm(centers.view(-1, num_features).t()).view(bs, num_classes, bag_size)

        prob = F.softmax(inputs, dim=-1)
        simClass = torch.sum(prob*inputs, dim=-1)

        lossClassify = F.cross_entropy(simClass, targets.long(), reduction="none") * target_weights

        return lossClassify.mean()