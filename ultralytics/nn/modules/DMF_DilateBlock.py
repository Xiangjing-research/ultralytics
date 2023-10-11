# YOLOv5 common modules

import torch
import torch.nn as nn

from models.dilateformer import DilateBlock


class DFMSDA(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention '''
    ''' 多模态差分特征 DilateAttention Cross-Modality Difference Feature  Dilateformer CMDF'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., kernel_size=3,
                 dilation=[1, 2, 3, 4],
                 cpe_per_block=False):
        super().__init__()

        self.dilateblock = DilateBlock(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation,
                                cpe_per_block=cpe_per_block)

    def forward(self, x):
        vi_feature, ir_feature = x[0], x[1]
        sub_vi_ir = vi_feature - ir_feature
        vi_ir_div = self.dilateblock(sub_vi_ir)

        # 特征加上各自的带有简易通道注意力机制的互补特征
        return torch.matmul(vi_feature, vi_ir_div), torch.matmul(vi_feature, vi_ir_div)

