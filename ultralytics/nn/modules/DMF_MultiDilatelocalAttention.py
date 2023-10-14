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

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 attn_drop=0., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, proj_drop=0.,
                 kernel_size=3, dilation=[1, 2, 3, 4]):
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

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    # def forward(self, qkv):
    #     B, _, H, W, C = qkv.shape
    #     qkv = qkv.reshape(B, _, H, W, self.num_dilation, C // self.num_dilation, ).permute(4, 1, 0, 5, 2, 3)
    #     # num_dilation,3,B,C//num_dilation,H,W
    #     x = Tensor(self.num_dilation, B, H, W, C // self.num_dilation).cuda()
    #     # num_dilation, B, H, W, C//num_dilation
    #     for i in range(self.num_dilation):
    #         x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
    #     x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, sub_feat, ori_feat):

        B, C, H, W = ori_feat.shape  # ([8, 96, 56, 56])
        q, k, v = self.q(sub_feat), self.k(ori_feat), self.v(ori_feat),
        q = q.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3,
                                                                                  4)  # num_dilation, B, C//num_dilation , H, W
        k = k.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        v = v.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 2, 3, 4)
        sub_feat = sub_feat.reshape(self.num_dilation, B, H, W, C // self.num_dilation)
        for i in range(self.num_dilation):
            sub_feat[i] = self.dilate_attention[i](q[i], k[i], v[i])  # B, H, W,C//num_dilation
        sub_feat = sub_feat.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        sub_feat = self.proj(sub_feat)
        sub_feat = self.proj_drop(sub_feat)
        sub_feat = sub_feat + self.drop_path(self.mlp(self.norm2(sub_feat)))
        sub_feat = sub_feat.permute(0, 3, 1, 2)
        return sub_feat


class DFMSDA(nn.Module):
    ''' 差分模态特征 DMF'''
    ''' Difference Feature MultiDilatelocalAttention  DFMDA'''
    ''' 多模态差分特征  Cross-Modality Difference Feature DilateAttention CMDFDA'''
    """ Self attention Layer"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., drop_path=0., attn_drop=0.,
                 kernel_size=3, dilation=[1, 2, 3, 4], norm_layer=nn.LayerNorm, act_layer=nn.GELU
                 ):
        super().__init__()

        self.attn_vi_ir = MultiDilatelocalAttention(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    attn_drop=attn_drop, drop=drop, drop_path=drop_path,
                                                    act_layer=act_layer,
                                                    norm_layer=norm_layer, kernel_size=kernel_size, dilation=dilation)
        self.attn_ir_vi = MultiDilatelocalAttention(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    attn_drop=attn_drop, drop=drop, drop_path=drop_path,
                                                    act_layer=act_layer,
                                                    norm_layer=norm_layer, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        vi_feature, ir_feature = x[0], x[1]
        sub_vi_ir = vi_feature - ir_feature
        vi_ir_div = self.attn_vi_ir(sub_vi_ir, ir_feature)

        sub_ir_vi = ir_feature - vi_feature
        ir_vi_div = self.attn_ir_vi(sub_ir_vi, vi_feature)

        # 特征加上各自的带有简易通道注意力机制的互补特征
        return vi_ir_div, ir_vi_div

    # def forward(self, x):
    #     vi_feature, ir_feature = x[0], x[1]  # (tensor): dim:(B, C, H, W)
    #     sub_vi_ir =torch.sub(vi_feature,ir_feature)
    #     vi_fea = torch.stack([sub_vi_ir, vi_feature, vi_feature], dim=1)  # (tensor): dim:(B, 3 ,C, H, W)
    #
    #     # if self.cpe_per_block:
    #     #     x = x + self.pos_embed(x)
    #     vi_fea = vi_fea.permute(0, 1, 3, 4, 2)  # (tensor): dim:(B, 3, H, W ,C)
    #     vi_fea = vi_fea + self.drop_path(self.attn_vi_ir(self.norm1(vi_fea)))
    #     vi_fea = vi_fea + self.drop_path(self.mlp(self.norm2(vi_fea)))
    #     vi_fea = vi_fea.permute(0, 3, 1, 2)
    #     # B, C, H, W
    #     #
    #     # sub_ir_vi = torch.sub(ir_feature,vi_feature)
    #     # vi_fea = torch.cat([sub_ir_vi, ir_feature, ir_feature], dim=1)  # (tensor): dim:(B, 3 ,C, H, W)
    #     #
    #     # # if self.cpe_per_block:
    #     # #     x = x + self.pos_embed(x)
    #     # vi_fea = vi_fea.permute(0, 1, 3, 4, 2)  # (tensor): dim:(B, 3, H, W ,C)
    #     # vi_fea = vi_fea + self.drop_path(self.attn_vi_ir(self.norm1(x)))
    #     # vi_fea = vi_fea + self.drop_path(self.mlp(self.norm2(x)))
    #     # vi_fea = vi_fea.permute(0, 3, 1, 2)
    #     # # B, C, H, W
    #
    #     # 特征加上各自的带有简易通道注意力机制的互补特征
    #     return torch.add(vi_feature, vi_fea), torch.add(ir_feature, vi_fea)
