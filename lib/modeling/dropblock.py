
import torch
import torch.nn.functional as F
from torch import nn


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    device = logits.device
    gumbels = sample_gumbel(logits.shape, device, eps)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class DropBlock(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, cfg):
        super(DropBlock, self).__init__()

        self.drop_prob = cfg.DROPBLOCK.DROP_PROB
        self.block_size = cfg.DROPBLOCK.BLOCK_SIZE

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            block_mask[:, 0, :] = 1.0
            block_mask[:, -1, :] = 1.0
            block_mask[:, :, 0] = 1.0
            block_mask[:, :, -1] = 1.0

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class BasicBlock(nn.Module):
    def __init__(self, planes, stride=1, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, 2 )
        self.bn2 = norm_layer(2)
        self.downsample = conv1x1(planes, 2)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, drop_prob):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)
        out += identity

        out_mask = torch.sigmoid(out[:,0:1]) * drop_prob
        out_bg = 1 - out_mask
        new_out = torch.cat((out_mask, out_bg), dim=1)

        return new_out


class ConvConcreteDB(torch.nn.Module):
    def __init__(self, cfg, planes):
        super(ConvConcreteDB, self).__init__()
        self.roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION  
        self.drop_prob = cfg.DROPBLOCK.DROP_PROB
        self.block_size = cfg.DROPBLOCK.BLOCK_SIZE
        self.tau = cfg.DROPBLOCK.GSM_THRES  
        self.conv = BasicBlock(planes) 
        self.is_hard = cfg.DROPBLOCK.IS_HARD

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x.detach())
            _scores = self.conv(x.detach(), gamma)

            # creat mask
            scores = gumbel_softmax(_scores.add(1e-10).log(), dim=1, tau=self.tau, hard=self.is_hard, eps=1e-10)
            mask = scores[:, 0]

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
