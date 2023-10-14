import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor

from ultralytics.nn.modules.dilateformer import DilateAttention, Mlp


class DFMSDA(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention '''
    ''' 多模态差分特征 DilateAttention Cross-Modality Difference Feature  Dilateformer CMDF'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., kernel_size=3,
                 dilation=[1, 2, 3, 4],
                 cpe_per_block=False):
        super().__init__()

        self.dilateblock1 = DilateBlock(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation,
                                 cpe_per_block=cpe_per_block)
        self.dilateblock2 = DilateBlock(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation,
                                 cpe_per_block=cpe_per_block)

    def forward(self, x):
        vi_feature, ir_feature = x[0], x[1]
        sub_vi_ir = vi_feature - ir_feature
        vi_ir_div = self.dilateblock1(sub_vi_ir, ir_feature, ir_feature)

        sub_ir_vi = ir_feature - vi_feature
        ir_vi_div = self.dilateblock2(sub_ir_vi, vi_feature, vi_feature)

        # 特征加上各自的带有简易通道注意力机制的互补特征
        return vi_ir_div, ir_vi_div


class DilateBlock(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention '''
    ''' 多模态差分特征 DilateAttention Cross-Modality Difference Feature  Dilateformer CMDF'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3, 4],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x1, x2, x3):
        # if self.cpe_per_block:
        #     x = x + self.pos_embed(x)
        x1, x2, x3 = x1.permute(0, 2, 3, 1), x2.permute(0, 2, 3, 1), x3.permute(0, 2, 3, 1)  # B, H, W, C
        x = x1 + self.drop_path(self.attn(self.norm1(x1), self.norm1(x2), self.norm1(x3)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        # B, C, H, W
        return x


class MultiDilatelocalAttention(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention '''
    ''' 多模态差分特征 DilateAttention Cross-Modality Difference Feature  Dilateformer CMDF'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        # self.qkv = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, x3):
        B, H, W, C = x1.shape  # ([8, 96, 56, 56])
        q, k, v = self.q(x1.permute(0, 3, 1, 2)), self.k(x2.permute(0, 3, 1, 2)), self.v(x3.permute(0, 3, 1, 2)),
        q = q.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        k = k.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        v = v.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        x = Tensor(self.num_dilation, B, H, W, C // self.num_dilation).cuda()
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](q[i], k[i], v[i])  # B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
