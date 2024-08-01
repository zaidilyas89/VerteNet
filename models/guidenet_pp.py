from .dec_net import DecNet_guidenet_pp
from . import resnet
import torch.nn as nn
import numpy as np
import timm_local
class guidenet_pp(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(guidenet_pp, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        # channels = [3, 64, 64, 128, 256, 512]
        channels = [3, 24, 48, 64, 160, 256]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = resnet.resnet34(pretrained=pretrained)
        self.base_network = timm_local.create_model('tf_efficientnetv2_s', pretrained=True, features_only=True)
        self.dec_net = DecNet_guidenet_pp(heads, final_kernel, head_conv, channels[self.l1])


    def forward(self, x):
        x1 = x
        x = self.base_network(x)
        x.insert(0, x1)
        dec_dict = self.dec_net(x)
        return dec_dict