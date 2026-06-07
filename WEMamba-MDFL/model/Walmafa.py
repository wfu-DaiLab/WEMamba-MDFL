import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import einops
from einops import rearrange
import numpy as np

from . import blocks
from mamba_ssm import Mamba
from src.fdconv import FDConv

m = None


def data_transform(X): return 2 * X - 1.0


def inverse_data_transform(X): return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__();
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False));
        self.add_module('bn', torch.nn.BatchNorm2d(b));
        torch.nn.init.constant_(self.bn.weight, bn_weight_init);
        torch.nn.init.constant_(self.bn.bias, 0)


def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__();
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]; return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x): sigma = x.var(-1, keepdim=True, unbiased=False); return x / torch.sqrt(
        sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.bias = nn.Parameter(torch.zeros(torch.Size(normalized_shape)))

    def forward(self, x): mu = x.mean(-1, keepdim=True); sigma = x.var(-1, keepdim=True, unbiased=False); return (
                                                                                                                             x - mu) / torch.sqrt(
        sigma + 1e-5) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__();
        hidden_features = int(dim * ffn_expansion_factor);
        self.project_in = nn.Conv2d(dim, hidden_features, 1, bias=bias);
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias);
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x): x = self.project_in(x); x = self.dwconv(x); x = F.gelu(x); x = self.project_out(x); return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False): super(OverlapPatchEmbed,
                                                                self).__init__(); self.proj = nn.Conv2d(in_c, embed_dim,
                                                                                                        3, 1, 1,
                                                                                                        bias=bias)

    def forward(self, x): return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat): super(Downsample, self).__init__(); self.body = nn.Sequential(
        nn.Conv2d(n_feat, n_feat * 2, 3, 2, 1, bias=False))

    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat): super(Upsample, self).__init__(); self.body = nn.Sequential(
        nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x): return self.body(x)


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
        self.channel_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
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


class HighFrequencyFDConvBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.fdconv_hl = FDConv(in_channels=dim, out_channels=dim, weight_gen_method="st")
        self.fdconv_lh = FDConv(in_channels=dim, out_channels=dim, weight_gen_method="st")
        self.fdconv_hh = FDConv(in_channels=dim, out_channels=dim, weight_gen_method="st")

    def forward(self, hl, lh, hh):
        enhanced_hl = self.fdconv_hl(hl)
        enhanced_lh = self.fdconv_lh(lh)
        enhanced_hh = self.fdconv_hh(hh)
        return torch.cat((enhanced_hl, enhanced_lh, enhanced_hh), dim=0)



class MDFL(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(MDFL, self).__init__()
        self.DWT = blocks.DWT()
        self.IWT = blocks.IWT()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.low_freq_processor = HybridAttentionMamba(dim=dim)
        self.high_freq_processor = HighFrequencyFDConvBlock(dim=dim, num_heads=num_heads)

    def forward(self, input_):
        x = input_
        x_norm = self.norm1(x)
        x_transformed = data_transform(x_norm)

        dwt_out = self.DWT(x_transformed)
        ll, hl, lh, hh = torch.chunk(dwt_out, 4, dim=0)

        processed_LL = self.low_freq_processor(ll)

        processed_high = self.high_freq_processor(hl, lh, hh)

        output_iwt = self.IWT(torch.cat((processed_LL, processed_high), dim=0))
        output = inverse_data_transform(output_iwt)

        x = x + output
        x = x + self.ffn(self.norm2(x))
        return x


class Walmafa(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=32,
                 num_blocks=[3, 4, 5],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 skip=False
                 ):
        super(Walmafa, self).__init__()

        dim_level1, dim_level2, dim_level3 = dim, dim * 2, dim * 4
        heads1, heads2, heads3 = heads[0], heads[1], heads[2]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim_level1)
        self.encoder_level1 = nn.Sequential(*[
            MDFL(dim=dim_level1, num_heads=heads1, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1 = Downsample(dim_level1)
        self.encoder_level2 = nn.Sequential(*[
            MDFL(dim=dim_level2, num_heads=heads2, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2 = Downsample(dim_level2)
        self.encoder_level3 = nn.Sequential(*[
            MDFL(dim=dim_level3, num_heads=heads3, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent = nn.Sequential(
            nn.Conv2d(dim_level3, dim_level3, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
        )
        self.up2 = Upsample(dim_level3)
        self.decoder_level2_conv = nn.Conv2d(dim_level2 + dim_level2, dim_level2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            MDFL(dim=dim_level2, num_heads=heads2,
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up1 = Upsample(dim_level2)
        self.decoder_level1_conv = nn.Conv2d(dim_level1 + dim_level1, dim_level1, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            MDFL(dim=dim_level1, num_heads=heads1,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.output = nn.Conv2d(dim_level1, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.skip = skip

    def forward(self, inp_img, mask):
        inp = torch.cat([inp_img, mask], dim=1)
        enc1 = self.patch_embed(inp)
        enc1 = self.encoder_level1(enc1)
        enc2 = self.down1(enc1)
        enc2 = self.encoder_level2(enc2)
        enc3 = self.down2(enc2)
        enc3 = self.encoder_level3(enc3)
        latent = self.latent(enc3)
        dec2 = self.up2(latent)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder_level2_conv(dec2)
        dec2 = self.decoder_level2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder_level1_conv(dec1)
        dec1 = self.decoder_level1(dec1)
        out = self.output(dec1)
        if self.skip:
            out = out + inp_img
        return out
