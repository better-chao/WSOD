import torch
from torch import nn
from torch.nn import functional as F
from core.config import cfg


class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

        self._init_weights()

    
    def _init_weights(self):
        pass

    def detectron_weight_mapping(self):
        if cfg.NONLOCAL.SUB_SAMPLE:
            detectron_weight_mapping = {
                'g.0.weight': 'g_0_w',
                'g.0.bias': 'g_0_b'
            }
        else:
            detectron_weight_mapping = {
                'g.weight': 'g_w',
                'g.bias': 'g_b'
            }

        if cfg.NONLOCAL.BN_LAYER:
            detectron_weight_mapping["W.0.weight"] = "W_0_weight"
            detectron_weight_mapping["W.0.bias"] = "W_0_bias"
            detectron_weight_mapping["W.1.running_mean"] = "W_1_running_mean"
            detectron_weight_mapping["W.1.running_var"] = "W_1_running_var"
            detectron_weight_mapping["W.1.num_batches_tracked"] = "W_1_num_batches_tracked"
            detectron_weight_mapping["W.1.weight"] = "W_1_weight"
            detectron_weight_mapping["W.1.bias"] = "W_1_bias"
        else:
            detectron_weight_mapping["W.weight"] = "W_weight"
            detectron_weight_mapping["W.bias"] = "W_bias"

        # if self.sub_sample:
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        

        g_x = g_x.permute(0, 2, 1).contiguous()
        # g_x.register_hook(lambda grad: grad.contiguous())
        # g_x = g_x.reshape(g_x.shape[0], g_x.shape[2], g_x.shape[1])
        # g_x = g_x.transpose(1, 2)
        

        theta_x = x.view(batch_size, self.in_channels, -1)
        
        theta_x = theta_x.permute(0, 2, 1).contiguous()
        # theta_x.register_hook(lambda grad: grad.contiguous())
        # theta_x = theta_x.reshape(theta_x.shape[0], theta_x.shape[2], theta_x.shape[1])
        # theta_x = theta_x.transpose(1, 2)
        

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
            
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        
        f_div_C = F.softmax(f, dim=-1)
        

        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        # y.register_hook(lambda grad: grad.contiguous())
        # y = y.reshape(y.shape[0], y.shape[2], y.shape[1])
        # y = y.transpose(1, 2).contiguous()
        
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        
        z = W_y + x

        return z


class GraphNeck(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

        self._init_weights()

    
    def _init_weights(self):
        pass

    def detectron_weight_mapping(self):
        if cfg.NONLOCAL.SUB_SAMPLE:
            detectron_weight_mapping = {
                'g.0.weight': 'g_0_w',
                'g.0.bias': 'g_0_b'
            }
        else:
            detectron_weight_mapping = {
                'g.weight': 'g_w',
                'g.bias': 'g_b'
            }

        if cfg.NONLOCAL.BN_LAYER:
            detectron_weight_mapping["W.0.weight"] = "W_0_weight"
            detectron_weight_mapping["W.0.bias"] = "W_0_bias"
            detectron_weight_mapping["W.1.running_mean"] = "W_1_running_mean"
            detectron_weight_mapping["W.1.running_var"] = "W_1_running_var"
            detectron_weight_mapping["W.1.num_batches_tracked"] = "W_1_num_batches_tracked"
            detectron_weight_mapping["W.1.weight"] = "W_1_weight"
            detectron_weight_mapping["W.1.bias"] = "W_1_bias"
        else:
            detectron_weight_mapping["W.weight"] = "W_weight"
            detectron_weight_mapping["W.bias"] = "W_bias"

        # if self.sub_sample:
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        

        g_x = g_x.permute(0, 2, 1).contiguous()
        

        theta_x = x.view(batch_size, self.in_channels, -1)
        
        theta_x = theta_x.permute(0, 2, 1).contiguous()
        

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
            
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        
        f_div_C = F.softmax(f, dim=-1)
        

        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        
        z = W_y + x

        return z

