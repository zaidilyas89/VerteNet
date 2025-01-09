import torch.nn as nn
import torch
from .model_parts import CombinationModule, CombinationModule_proposed

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super(DecNet, self).__init__()
        self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
        self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
    
############################################################################

# class DecNet_guidenet_pp(nn.Module):
#     def __init__(self, heads, final_kernel, head_conv, channel):
#         super(DecNet_guidenet_pp, self).__init__()
#         self.dec_c2 = CombinationModule(64, 48, batch_norm=True)
#         self.dec_c3 = CombinationModule(160, 64, batch_norm=True)
#         self.dec_c4 = CombinationModule(256, 160, batch_norm=True)
#         self.heads = heads
#         for head in self.heads:
#             classes = self.heads[head]
#             if head == 'wh':
#                 fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
#             else:
#                 fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
#                                              padding=final_kernel // 2, bias=True))
#             if 'hm' in head:
#                 fc[-1].bias.data.fill_(-2.19)
#             else:
#                 self.fill_fc_weights(fc)

#             self.__setattr__(head, fc)


#     def fill_fc_weights(self, layers):
#         for m in layers.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)


#     def forward(self, x):
#         c4_combine = self.dec_c4(x[-1], x[-2])
#         c3_combine = self.dec_c3(c4_combine, x[-3])
#         c2_combine = self.dec_c2(c3_combine, x[-4])
#         dec_dict = {}
#         for head in self.heads:
#             dec_dict[head] = self.__getattr__(head)(c2_combine)
#             if 'hm' in head:
#                 dec_dict[head] = torch.sigmoid(dec_dict[head])
#         return dec_dict
    
    
    
class DecNet_vertenet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super(DecNet_vertenet, self).__init__()
        self.dec_c2 = CombinationModule_proposed(64, 48, batch_norm=True, patch_size = 8, dim = 256*128)
        self.dec_c3 = CombinationModule_proposed(160, 64, batch_norm=True, patch_size = 8, dim = 128*64)
        self.dec_c4 = CombinationModule_proposed(256, 160, batch_norm=True, patch_size = 8, dim = 64*32)
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                              padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict    
