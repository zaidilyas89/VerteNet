import torch.nn as nn
import torch.nn.functional as F
import torch
# from pdb import set_trace as stx
import numbers
import math
from einops import rearrange


class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((x_up, x_low), 1))


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def get_patches(image, patch_size):
    # image = image.transpose(0,2,3,1)
    image = rearrange(image, 'b c h w -> b h w c')
    b, h, w, c = image.shape
    patches = rearrange(image,
                       'b (h_patch h) (w_patch w) c -> b (h_patch w_patch) h w c',
                       h_patch=patch_size[0], w_patch=patch_size[1], h=h // patch_size[0], w=w // patch_size[1])
    return patches


def reconstruct_image(patches, image_shape, patch_size):
    b, h, w, c = image_shape
    image = rearrange(patches,
                       'b (h_patch w_patch) h w c -> b (h_patch h) (w_patch w) c',
                       h_patch=patch_size[0], w_patch=patch_size[1], h=h // patch_size[0], w=w // patch_size[1])
    image = rearrange(image, 'b h w c -> b c h w')
    return image


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # h, w = x.shape[-2:]
        return self.body(x)

class LayerNorm_Channel(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Channel, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class HiLo_CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, channels, patch_size):
        super(HiLo_CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.patch_size = patch_size
        # self.qkv = nn.Conv2d(81, 81*3, kernel_size=1, bias=bias)
        self.q_Hi = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.k_Hi = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.v_Hi = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.q_Lo = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.k_Lo = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.v_Lo = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.convTranspose = nn.ConvTranspose2d(channels, channels, 2, 2)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(channels*2, channels, kernel_size=1, bias=bias)
        # self.project_out = nn.Linear(dim,dim, bias=False)
        


    def forward(self, x_up, x_down):

        b_down,c_down,h_down,w_down = x_down.shape
        x_down_Hi = x_down
        x_down_Lo = self.downsample(x_down)

        b_up,c_up,h_up,w_up = x_up.shape
        x_up_Hi = x_up
        x_up_Lo = self.downsample(x_up)
        
        
        # Patches generation from x_down (decoder end)
        b_down_Hi, c_down_Hi, h_down_Hi, w_down_Hi= x_down_Hi.shape
        image_shape_out_Hi = b_down_Hi, h_down_Hi, w_down_Hi, c_down_Hi
        patches_x_down_Hi = get_patches(x_down_Hi, (self.patch_size,self.patch_size))
        b_patches_x_down_Hi, patch_square_patches_x_down_Hi, h_patches_x_down_Hi, w_patches_x_down_Hi, c_patches_x_down_Hi = patches_x_down_Hi.shape
        patches_x_down_Hi = rearrange(patches_x_down_Hi, 'b patches h w c -> b (h w) c patches')
        
        
        
        b_down_Lo, c_down_Lo, h_down_Lo, w_down_Lo= x_down_Lo.shape
        image_shape_out_Lo = b_down_Lo, h_down_Lo, w_down_Lo, c_down_Lo
        patches_x_down_Lo = get_patches(x_down_Lo, (self.patch_size,self.patch_size))
        b_patches_x_down_Lo, patch_square_patches_x_down_Lo, h_patches_x_down_Lo, w_patches_x_down_Lo, c_patches_x_down_Lo = patches_x_down_Lo.shape
        patches_x_down_Lo = rearrange(patches_x_down_Lo, 'b patches h w c -> b (h w) c patches')

        
        # Query Generation
        q_down_Hi = self.q_Hi(patches_x_down_Hi)
        q_down_Lo = self.q_Lo(patches_x_down_Lo)        
        
        
        # Patches generation from x_up (encoder end)        
        b_up_Hi, c_up_Hi, h_up_Hi, w_up_Hi= x_up_Hi.shape
        patches_x_up_Hi = get_patches(x_up_Hi, (self.patch_size,self.patch_size))
        b_patches_x_up_Hi, patch_square_patches_x_up_Hi, h_patches_x_up_Hi, w_patches_x_up_Hi, c_patches_x_up_Hi = patches_x_up_Hi.shape
        patches_x_up_Hi = rearrange(patches_x_up_Hi, 'b patches h w c -> b (h w) c patches')

        b_up_Lo, c_up_Lo, h_up_Lo, w_up_Lo= x_up_Lo.shape
        patches_x_up_Lo = get_patches(x_up_Lo, (self.patch_size,self.patch_size))
        b_patches_x_up_Lo, patch_square_patches_x_up_Lo, h_patches_x_up_Lo, w_patches_x_up_Lo, c_patches_x_up_Lo = patches_x_up_Lo.shape
        patches_x_up_Lo = rearrange(patches_x_up_Lo, 'b patches h w c -> b (h w) c patches')        
        
        
        k_up_Hi = self.k_Hi(patches_x_up_Hi)   
        v_up_Hi = self.v_Hi(patches_x_up_Hi)  
        
        k_up_Lo = self.k_Lo(patches_x_up_Lo)   
        v_up_Lo = self.v_Lo(patches_x_up_Lo) 
        
        q_down_Hi = rearrange(q_down_Hi, 'b hw c patches -> b hw patches c')
        q_down_Lo = rearrange(q_down_Lo, 'b hw c patches -> b hw patches c')
        
        k_up_Hi = rearrange(k_up_Hi, 'b hw c patches -> b hw patches c')
        k_up_Lo = rearrange(k_up_Lo, 'b hw c patches -> b hw patches c')    
        
        v_up_Hi = rearrange(v_up_Hi, 'b hw c patches -> b hw patches c')
        v_up_Lo = rearrange(v_up_Lo, 'b hw c patches -> b hw patches c') 
        
        q_down_Hi = torch.nn.functional.normalize(q_down_Hi, dim=-1)
        q_down_Lo = torch.nn.functional.normalize(q_down_Lo, dim=-1)
        
        k_up_Hi = torch.nn.functional.normalize(k_up_Hi, dim=-1)
        k_up_Lo = torch.nn.functional.normalize(k_up_Lo, dim=-1)

        attn_Hi = (q_down_Hi @ k_up_Hi.transpose(-2, -1)) * self.temperature
        attn_Hi = attn_Hi.softmax(dim=-1)

        out_Hi = (attn_Hi @ v_up_Hi)
        
        
        attn_Lo = (q_down_Lo @ k_up_Lo.transpose(-2, -1)) * self.temperature
        attn_Lo = attn_Lo.softmax(dim=-1)

        out_Lo = (attn_Lo @ v_up_Lo)
        
        out_Hi = rearrange(out_Hi, 'b hw patches c -> b hw c patches')
        out_Hi = rearrange(out_Hi, 'b (h w) c patches -> b patches h w c', h = h_patches_x_down_Hi, w= w_patches_x_down_Hi)

        out_Lo = rearrange(out_Lo, 'b hw patches c -> b hw c patches')
        out_Lo = rearrange(out_Lo, 'b (h w) c patches -> b patches h w c', h = h_patches_x_down_Lo, w= w_patches_x_down_Lo)        
        
        
        
        out_Hi = reconstruct_image(out_Hi, image_shape_out_Hi, (self.patch_size,self.patch_size))
        out_Lo = reconstruct_image(out_Lo, image_shape_out_Lo, (self.patch_size,self.patch_size))
        out_Lo = self.convTranspose(out_Lo)
        
        out = torch.cat((out_Hi,out_Lo),1)
        out = self.project_out(out)

        return out 
    
    
class HiLo_SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, channels, patch_size):
        super(HiLo_SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.patch_size = patch_size
        # self.qkv = nn.Conv2d(81, 81*3, kernel_size=1, bias=bias)
        self.q_Hi = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.k_Hi = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.v_Hi = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.q_Lo = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.k_Lo = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        self.v_Lo = nn.Linear(patch_size*patch_size,patch_size*patch_size, bias=False)
        
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.convTranspose = nn.ConvTranspose2d(channels, channels, 2, 2)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(channels*2, channels, kernel_size=1, bias=bias)
        # self.project_out = nn.Linear(dim,dim, bias=False)
        


    def forward(self, x):

        b,c,h,w = x.shape
        x_Hi = x
        x_Lo = self.downsample(x)

        # b_up,c_up,h_up,w_up = x_up.shape
        # x_up_Hi = x_up
        # x_up_Lo = self.downsample(x_up)
        
        
        # Patches generation from x_down (decoder end)
        b_Hi, c_Hi, h_Hi, w_Hi= x_Hi.shape
        image_shape_out_Hi = b_Hi, h_Hi, w_Hi, c_Hi
        patches_x_Hi = get_patches(x_Hi, (self.patch_size,self.patch_size))
        b_patches_x_Hi, patch_square_patches_x_Hi, h_patches_x_Hi, w_patches_x_Hi, c_patches_x_Hi = patches_x_Hi.shape
        patches_x_Hi = rearrange(patches_x_Hi, 'b patches h w c -> b (h w) c patches')
        
        
        
        b_Lo, c_Lo, h_Lo, w_Lo= x_Lo.shape
        image_shape_out_Lo = b_Lo, h_Lo, w_Lo, c_Lo
        patches_x_Lo = get_patches(x_Lo, (self.patch_size,self.patch_size))
        b_patches_x_Lo, patch_square_patches_x_Lo, h_patches_x_Lo, w_patches_x_Lo, c_patches_x_Lo = patches_x_Lo.shape
        patches_x_Lo = rearrange(patches_x_Lo, 'b patches h w c -> b (h w) c patches')

        
        # Query Generation
        q_Hi = self.q_Hi(patches_x_Hi)
        q_Lo = self.q_Lo(patches_x_Lo)        
        
        
        # Patches generation from x_up (encoder end)        
        b_Hi, c_Hi, h_Hi, w_Hi= x_Hi.shape
        patches_x_Hi = get_patches(x_Hi, (self.patch_size,self.patch_size))
        b_patches_x_Hi, patch_square_patches_x_Hi, h_patches_x_Hi, w_patches_x_Hi, c_patches_x_Hi = patches_x_Hi.shape
        patches_x_Hi = rearrange(patches_x_Hi, 'b patches h w c -> b (h w) c patches')

        b_Lo, c_Lo, h_Lo, w_Lo= x_Lo.shape
        patches_x_Lo = get_patches(x_Lo, (self.patch_size,self.patch_size))
        b_patches_x_Lo, patch_square_patches_x_Lo, h_patches_x_Lo, w_patches_x_Lo, c_patches_x_Lo = patches_x_Lo.shape
        patches_x_Lo = rearrange(patches_x_Lo, 'b patches h w c -> b (h w) c patches')        
        
        
        k_Hi = self.k_Hi(patches_x_Hi)   
        v_Hi = self.v_Hi(patches_x_Hi)  
        
        k_Lo = self.k_Lo(patches_x_Lo)   
        v_Lo = self.v_Lo(patches_x_Lo) 
        
        q_Hi = rearrange(q_Hi, 'b hw c patches -> b hw patches c')
        q_Lo = rearrange(q_Lo, 'b hw c patches -> b hw patches c')
        
        k_Hi = rearrange(k_Hi, 'b hw c patches -> b hw patches c')
        k_Lo = rearrange(k_Lo, 'b hw c patches -> b hw patches c')    
        
        v_Hi = rearrange(v_Hi, 'b hw c patches -> b hw patches c')
        v_Lo = rearrange(v_Lo, 'b hw c patches -> b hw patches c') 
        
        q_Hi = torch.nn.functional.normalize(q_Hi, dim=-1)
        q_Lo = torch.nn.functional.normalize(q_Lo, dim=-1)
        
        k_Hi = torch.nn.functional.normalize(k_Hi, dim=-1)
        k_Lo = torch.nn.functional.normalize(k_Lo, dim=-1)

        attn_Hi = (q_Hi @ k_Hi.transpose(-2, -1)) * self.temperature
        attn_Hi = attn_Hi.softmax(dim=-1)

        out_Hi = (attn_Hi @ v_Hi)
        
        
        attn_Lo = (q_Lo @ k_Lo.transpose(-2, -1)) * self.temperature
        attn_Lo = attn_Lo.softmax(dim=-1)

        out_Lo = (attn_Lo @ v_Lo)
        
        out_Hi = rearrange(out_Hi, 'b hw patches c -> b hw c patches')
        out_Hi = rearrange(out_Hi, 'b (h w) c patches -> b patches h w c', h = h_patches_x_Hi, w= w_patches_x_Hi)

        out_Lo = rearrange(out_Lo, 'b hw patches c -> b hw c patches')
        out_Lo = rearrange(out_Lo, 'b (h w) c patches -> b patches h w c', h = h_patches_x_Lo, w= w_patches_x_Lo)        
        
        
        
        out_Hi = reconstruct_image(out_Hi, image_shape_out_Hi, (self.patch_size,self.patch_size))
        out_Lo = reconstruct_image(out_Lo, image_shape_out_Lo, (self.patch_size,self.patch_size))
        out_Lo = self.convTranspose(out_Lo)
        
        out = torch.cat((out_Hi,out_Lo),1)
        out = self.project_out(out)

        return out 
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv = nn.Conv2d(81, 81*3, kernel_size=1, bias=bias)
        self.q = nn.Linear(dim,dim, bias=False)
        self.k = nn.Linear(dim,dim, bias=False)
        self.v = nn.Linear(dim,dim, bias=False)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Linear(dim,dim, bias=False)
        


    def forward(self, x):
        # b,c,h,w = x.shape
        # qkv = self.qkv_dwconv(self.qkv(x))
        # x1=x
        # x = rearrange(x, 'b c h w -> b c (h w)')
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b feature seq -> b seq feature')
        k = rearrange(k, 'b feature seq -> b seq feature')
        v = rearrange(v, 'b feature seq -> b seq feature')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
       
        out = rearrange(out, 'b seq feature -> b feature seq ')
        out = self.project_out(out)
        # out = rearrange(out, 'b c (h w) -> b c h w', h=9,w=9)
        return out
class TransformerBlock(nn.Module):
    def __init__(self, dim=None, head_size=16, channels=None, num_heads=None, ffn_expansion_factor=None, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.head_size = head_size
        self.channels = channels
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)', h = h, w = w)
        x = self.norm1(x)
        # x = rearrange(x, 'b c (h w) -> b c h w', h = 9, w = 9)
        # x = rearrange(x, 'b c (h w)-> b c h w ', h = h, w = w)
        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))
        x = rearrange(x, 'b c (h w) -> b c h w', h = h, w = w)
        return x

class DRCA_HiLoTransformerBlock(nn.Module):
    def __init__(self, dim=None, patch_size=8, channels=None, num_heads=None, ffn_expansion_factor=None, bias=False, LayerNorm_type='WithBias'):
        super(DRCA_HiLoTransformerBlock, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.channels = channels
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HiLo_CrossAttention(dim, num_heads, bias, channels, self.patch_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)
        
    def forward(self, x_up,x_low):
        b,c,h,w = x_up.shape
        x_up = rearrange(x_up, 'b c h w -> b c (h w)', h = h, w = w)
        x_up = self.norm1(x_up)
        # x = rearrange(x, 'b c (h w) -> b c h w', h = 9, w = 9)
        x_up = rearrange(x_up, 'b c (h w)-> b c h w ', h = h, w = w)

        b,c,h,w = x_low.shape
        x_low = rearrange(x_low, 'b c h w -> b c (h w)', h = h, w = w)
        x_low = self.norm1(x_low)
        # x = rearrange(x, 'b c (h w) -> b c h w', h = 9, w = 9)
        x_low = rearrange(x_low, 'b c (h w)-> b c h w ', h = h, w = w)

        
        x_low = x_low + self.attn(x_up,x_low)
        x_low = rearrange(x_low, 'b c h w-> b c (h w) ', h = h, w = w)
        x_low = self.norm2(x_low)
        x_low = rearrange(x_low, 'b c (h w)  -> b c h w ', h = h, w = w)
        x_low = x_low + self.ffn(x_low)
        # x_low = rearrange(x_low, 'b c (h w) -> b c h w', h = h, w = w)
        return x_low
        
        return

class DRSA_HiLoTransformerBlock(nn.Module):
    def __init__(self, dim=None, patch_size=8, channels=None, num_heads=None, ffn_expansion_factor=None, bias=False, LayerNorm_type='WithBias'):
        super(DRSA_HiLoTransformerBlock, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.channels = channels
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HiLo_SelfAttention(dim, num_heads, bias, channels, self.patch_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)', h = h, w = w)
        x = self.norm1(x)
        # x = rearrange(x, 'b c (h w) -> b c h w', h = 9, w = 9)
        x = rearrange(x, 'b c (h w)-> b c h w ', h = h, w = w)

        # b,c,h,w = x_low.shape
        # x_low = rearrange(x_low, 'b c h w -> b c (h w)', h = h, w = w)
        # x_low = self.norm1(x_low)
        # x = rearrange(x, 'b c (h w) -> b c h w', h = 9, w = 9)
        # x_low = rearrange(x_low, 'b c (h w)-> b c h w ', h = h, w = w)

        
        x = x + self.attn(x)
        x = rearrange(x, 'b c h w-> b c (h w) ', h = h, w = w)
        x = self.norm2(x)
        x = rearrange(x, 'b c (h w)  -> b c h w ', h = h, w = w)
        x = x + self.ffn(x)
        # x_low = rearrange(x_low, 'b c (h w) -> b c h w', h = h, w = w)
        return x
        
        return

class Channel_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Channel_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
class Channel_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Channel_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Channel_Transformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Channel_Transformer, self).__init__()

        self.norm1 = LayerNorm_Channel(dim, LayerNorm_type)
        self.attn = Channel_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm_Channel(dim, LayerNorm_type)
        self.ffn = Channel_FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CombinationModule_proposed(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False, patch_size = 8, dim = None):
        super(CombinationModule_proposed, self).__init__()
        # self.TransformerBlock = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        
        # self.t_s_block1 = TransformerBlock(dim=8*8, head_size=10, channels=c_up, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # self.t_s_block2 = TransformerBlock(dim=10*10, head_size=10, channels=512, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        
        # self.t_hilo_block2 = HiLoTransformerBlock(dim=20*20, channels=96, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.t_drsa_decoder_block_ = DRSA_HiLoTransformerBlock(dim=dim, patch_size=patch_size, channels=c_up, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.t_drsa_skip_block_ = DRSA_HiLoTransformerBlock(dim=dim, patch_size=patch_size, channels=c_up, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.t_drca_decoder_block_ = DRCA_HiLoTransformerBlock(dim=dim, patch_size=patch_size, channels=c_up, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.t_drca_skip_block_ = DRCA_HiLoTransformerBlock(dim=dim, patch_size=patch_size, channels=c_up, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        
        self.channel_transformer = Channel_Transformer(dim=c_up, num_heads = 2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        x_low = self.t_drsa_decoder_block_(x_low)
        x_up = self.t_drsa_skip_block_(x_up)
        
        x_low_ = self.t_drca_decoder_block_(x_up,x_low)
        x_up_ = self.t_drca_skip_block_(x_low,x_up)
        x = self.cat_conv(torch.cat((x_up_, x_low_), 1))
        x = self.channel_transformer(x)
        return x 

