import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.out_channels * 2,
                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()


        device = x.device
        x_cpu = x.detach().cpu().float()
        ffted_cpu = torch.fft.rfft2(x_cpu, dim=(-2, -1))
        ffted = ffted_cpu.to(device)

        scale = math.sqrt(h * w)
        ffted = ffted / scale


        ffted = torch.view_as_real(ffted)


        ffted = ffted.permute(0, 1, 4, 2, 3).reshape(batch, c * 2, h, w // 2 + 1).contiguous()


        ffted = self.conv(ffted)
        ffted = self.relu(self.bn(ffted))


        ffted = ffted.view(batch, self.out_channels, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)


        ffted_cpu = ffted.detach().cpu()
        output_cpu = torch.fft.irfft2(ffted_cpu, s=(h, w), dim=(-2, -1))
        output = output_cpu.to(device)
        output = output * scale

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        if self.enable_lfu:

            self.lfu = FourierUnit(out_channels // 4, out_channels // 2, groups)

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            sh, sw = h // split_no, w // split_no


            xs = x[:, :c // 2, :].reshape(n, c // 2, split_no, sh, split_no, sw)

            xs = xs.permute(0, 2, 4, 1, 3, 5).reshape(n * split_no * split_no, c // 2, sh, sw).contiguous()


            xs = self.lfu(xs)


            xs = xs.reshape(n, split_no, split_no, c, sh, sw)
            xs = xs.permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w).contiguous()
        else:
            xs = 0


        output = self.conv2(x + output + xs)
        return output


class FDConv(nn.Module):
    def __init__(self, in_channels, out_channels, dw_stride=1, dw_kernel_size=3,
                 num_heads=8,
                 weight_gen_method="st",
                 use_pe=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dw_stride = dw_stride
        self.dw_kernel_size = dw_kernel_size
        self.num_heads = num_heads
        self.use_pe = use_pe

        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel_size,
                                 stride=dw_stride, padding=(dw_kernel_size - 1) // 2,
                                 groups=in_channels, bias=False)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.head_dim = in_channels // num_heads
        self.fc = nn.Conv2d(self.head_dim, self.head_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.weight_gen_method = weight_gen_method

        if weight_gen_method == "st":
            self.filter_gen = SpectralTransform(in_channels, in_channels, stride=1, groups=1)
        else:
            self.filter_gen = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        if use_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, in_channels, 1, 1))

    def forward(self, x):
        if self.use_pe:
            x = x + self.pos_embed

        filter = self.filter_gen(x)
        n, c, h, w = x.shape


        x_grouped = x.reshape(n, self.num_heads, self.head_dim, h, w)
        filter_grouped = filter.reshape(n, self.num_heads, self.head_dim, h, w)

        pooled_x = x_grouped.mean(dim=(-1, -2), keepdim=True)
        pooled_x = pooled_x.view(n * self.num_heads, self.head_dim, 1, 1)

        weights = self.fc(pooled_x)
        weights = weights.view(n, self.num_heads, self.head_dim, 1, 1).transpose(1, 2)
        weights = self.softmax(weights)
        weights = weights.transpose(1, 2)

        filtered_x = (filter_grouped * weights).view(n, c, h, w)

        out = self.dw_conv(filtered_x)
        out = self.pw_conv(out)

        return out