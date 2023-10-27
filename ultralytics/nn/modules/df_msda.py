# YOLOv5 common modules

import torch.nn as nn
from torch import Tensor

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,HW,1,d
        k = self.unfold(k)
        k = k.reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,HW,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention '''
    ''' 多模态差分特征 DilateAttention Cross-Modality Difference Feature  Dilateformer CMDF'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_size=3,
                 dilation=None):
        super().__init__()
        head_dim = dim // num_heads
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

    def forward(self, q_mtrix, k_mtrix, v_mtrix):
        B, H, W, C = q_mtrix.shape  # ([8, 96, 56, 56])
        q_mtrix = q_mtrix.permute(0, 3, 1, 2)  # B, C, H, W
        k_mtrix = k_mtrix.permute(0, 3, 1, 2)  # B, C, H, W
        v_mtrix = v_mtrix.permute(0, 3, 1, 2)  # B, C, H, W

        q, k, v = self.q(q_mtrix), self.k(k_mtrix), self.v(v_mtrix),
        q = q.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3,
                                                                                  4)  # num_dilation, B, C//num_dilation , H, W
        k = k.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        v = v.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        q_mtrix = q_mtrix.reshape(self.num_dilation, B, H, W, C // self.num_dilation)
        for i in range(self.num_dilation):
            q_mtrix[i] = self.dilate_attention[i](q[i], k[i], v[i])  # B, H, W,C//num_dilation
        q_mtrix = q_mtrix.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        q_mtrix = self.proj(q_mtrix)
        q_mtrix = self.proj_drop(q_mtrix)
        return q_mtrix


# class DFMSDABlock(nn.Module):
#     ''' 差分模态特征 DMF'''
#     ''' Difference Feature MultiDilatelocalAttention  DFMDA'''
#     ''' 多模态差分特征  Cross-Modality Difference Feature DilateAttention CMDFDA'''
#     """ Self attention Layer"""
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3, 4]):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.norm1 = norm_layer(dim)
#         self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                               attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
#         # self.attn_ir_vi = MultiDilatelocalAttention(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#         #                                             qk_scale=qk_scale,
#         #                                             attn_drop=attn_drop, drop=drop, drop_path=drop_path,
#         #                                             act_layer=act_layer,
#         #                                             norm_layer=norm_layer, kernel_size=kernel_size, dilation=dilation)
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         vi_feature, ir_feature = x[0], x[1]
#         vi_feature = vi_feature.permute(0, 2, 3, 1)  # B, H, W, C
#         ir_feature = ir_feature.permute(0, 2, 3, 1)  # B, H, W, C
#         sub_vi_ir = vi_feature - ir_feature
#
#         sub_vi_ir = sub_vi_ir + self.drop_path(
#             self.attn(self.norm1(sub_vi_ir), self.norm1(vi_feature), self.norm1(vi_feature)))
#         sub_vi_ir = sub_vi_ir + self.drop_path(self.mlp(self.norm2(sub_vi_ir)))
#
#         sub_ir_vi = ir_feature - vi_feature
#         sub_ir_vi = sub_ir_vi + self.drop_path(
#             self.attn(self.norm1(sub_ir_vi), self.norm1(ir_feature), self.norm1(ir_feature)))
#         sub_ir_vi = sub_ir_vi + self.drop_path(self.mlp(self.norm2(sub_ir_vi)))
#
#         return sub_vi_ir.permute(0, 3, 1, 2), sub_ir_vi.permute(0, 3, 1, 2) # B, C, H, W

class DFMSDABlock(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention  DFMDA'''
    ''' 多模态差分特征  Cross-Modality Difference Feature DilateAttention CMDFDA'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm1_vi_ir = norm_layer(dim)
        self.norm1_ir_vi = norm_layer(dim)
        self.attn_vi_ir = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        self.attn_ir_vi = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_vi_ir = norm_layer(dim)
        self.norm2_ir_vi = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        vi_feature, ir_feature = x[0], x[1]
        vi_feature = vi_feature.permute(0, 2, 3, 1)  # B, H, W, C
        ir_feature = ir_feature.permute(0, 2, 3, 1)  # B, H, W, C
        sub_vi_ir = vi_feature - ir_feature

        sub_vi_ir = sub_vi_ir + self.drop_path(
            self.attn_vi_ir(self.norm1_vi_ir(sub_vi_ir), self.norm1_vi_ir(vi_feature), self.norm1_vi_ir(vi_feature)))
        sub_vi_ir = sub_vi_ir + self.drop_path(self.mlp(self.norm2_vi_ir(sub_vi_ir)))

        sub_ir_vi = ir_feature - vi_feature
        sub_ir_vi = sub_ir_vi + self.drop_path(
            self.attn_ir_vi(self.norm1_ir_vi(sub_ir_vi), self.norm1_ir_vi(ir_feature), self.norm1_ir_vi(ir_feature)))
        sub_ir_vi = sub_ir_vi + self.drop_path(self.mlp2(self.norm2_ir_vi(sub_ir_vi)))

        return sub_vi_ir.permute(0, 3, 1, 2), sub_ir_vi.permute(0, 3, 1, 2) # B, C, H, W