import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_sa = self.conv1(x_cat)
        return self.sigmoid(x_sa)


class HybridAttentionMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channel_mamba = Mamba(
            d_model=dim,
            d_state=32,
            d_conv=4,
            expand=2
        )
        self.spatial_attention = SpatialAttention()
        self.ln = nn.LayerNorm(normalized_shape=dim)

    def forward(self, x):

        b, c, h, w = x.shape
        x_res = x

        x_seq = x.reshape(b, c, -1).permute(0, 2, 1)
        x_norm = self.ln(x_seq)
        x_channel_processed = self.channel_mamba(x_norm)

        x_channel_att = x_channel_processed.permute(0, 2, 1).reshape(b, c, h, w)

        x_spatial_att_map = self.spatial_attention(x)
        x_spatial_enhanced = x_channel_att * x_spatial_att_map

        return x_res + x_spatial_enhanced
