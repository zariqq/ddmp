import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch import tensor
from math import log as m_log

from config import T, image_shape


class TimeEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

        pe = TimeEmbedding.positional_encodings(T, d_in)
        self.register_buffer("pe", pe)

    def positional_encodings(T, dim):
        expon = torch.arange(0, dim, 2) / dim
        pos = torch.arange(0, T)[:, None]
        pe_arg = torch.exp(-expon * m_log(10_000))
        pe = torch.empty(T, dim)

        pe[:, ::2] = torch.sin(pos * pe_arg)
        pe[:, 1::2] = torch.cos(pos * pe_arg)

        return pe

    def forward(self, t):
        enc = self.pe[t]
        return self.net(enc)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_groups):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.qkv_w = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.scale = channels ** (-0.5)

        # this need to resnet learn without attention
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        residual = x
        B, C, H, W = x.shape
        N = H * W

        h = self.norm(x)
        qkv = self.qkv_w(h)  # (B, 3 * C, H, W)
        qkv = qkv.view(B, 3 * C, -1).permute(0, 2, 1)  # (B, N, 3 * C)
        q, k, v = qkv.chunk(3, dim=2)  # q, k, v: (B, N, C)

        attn_score = torch.matmul(q, k.transpose(-2, -1))  # (B, N, N)
        attn = func.softmax(attn_score * self.scale, dim=-1)
        res = torch.matmul(attn, v).permute(0, 2, 1)  # (B, N, C) -> (B, C, N)

        proj = self.proj_out(res.view(B, C, H, W))

        return proj + residual


"""
TODO: implement wide ResNet block
"""


class WideResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        dropout=0.1,
        in_num_groups: int = None,
        out_num_groups: int = None,
    ):
        super().__init__()

        in_num_groups = in_num_groups if in_num_groups is not None else in_channels
        out_num_groups = out_num_groups if out_num_groups is not None else out_channels

        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        if in_channels == out_channels:
            self.proj_res = nn.Identity()
        else:
            self.proj_res = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )

        self.backbone_first = nn.Sequential(
            nn.GroupNorm(
                num_groups=in_num_groups, num_channels=in_channels
            ),  # (B, Cin, H, W)
            nn.SiLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),  # (B, Cout, H, W)
        )
        self.backbone_second = nn.Sequential(
            nn.GroupNorm(num_groups=out_num_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, time_emb):
        residual = self.proj_res(x)
        bb1 = self.backbone_first(x)  # (B, Cout, H, W)
        acc_time = bb1 + self.time_proj(time_emb)[:, :, None, None]
        bb2 = self.backbone_second(acc_time)
        return bb2 + residual


"""
TODO: implement U-Net (the whole model with Wide ResNet and Attention blocks)
"""


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        time_emb_dim,
        in_num_groups,
        out_channels=None,
        dropout=0.1,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        self.resnet = WideResNet(
            in_channels,
            out_channels,
            time_emb_dim,
            dropout,
            in_num_groups,
            in_num_groups,
        )
        self.attn = SelfAttention(out_channels, in_num_groups)

    def forward(self, x, time_emb):
        h = self.resnet(x, time_emb)

        return self.attn(h)


from dataclasses import dataclass


@dataclass
class Hypers:
    time_emb_dim: int


class UNet(nn.Module):
    def __init__(self, hyp: Hypers):
        super().__init__()

        base_ch = 32
        channel_mults = [2, 4, 8, 16]
        num_groups = 32
        time_emb_dim = hyp.time_emb_dim

        self.time_emb = TimeEmbedding(time_emb_dim, time_emb_dim)

        # input / output projections
        self.start_conv = nn.Conv2d(image_shape[0], base_ch, kernel_size=3, padding=1)
        self.end_conv = nn.Conv2d(base_ch, image_shape[0], kernel_size=3, padding=1)

        # encoder
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        skip_channels = []
        in_ch = base_ch
        for mult in channel_mults:
            out_ch = base_ch * mult
            self.encoder.append(Block(in_ch, time_emb_dim, num_groups))
            self.downsamples.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            skip_channels.append(in_ch)

            in_ch = out_ch

        # bottleneck
        self.bottleneck_res1 = WideResNet(
            in_ch,
            in_ch,
            time_emb_dim,
            in_num_groups=num_groups,
            out_num_groups=num_groups,
        )
        self.bottleneck_attn = SelfAttention(in_ch, num_groups)
        self.bottleneck_res2 = WideResNet(
            in_ch,
            in_ch,
            time_emb_dim,
            in_num_groups=num_groups,
            out_num_groups=num_groups,
        )

        # decoder
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_ch = in_ch
        for skip_ch in reversed(skip_channels):
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current_ch, skip_ch, kernel_size=3, padding=1),
                )
            )
            self.decoder.append(
                Block(skip_ch * 2, time_emb_dim, num_groups, out_channels=skip_ch)
            )
            current_ch = skip_ch

    def forward(self, x, t):
        h = self.start_conv(x)  # (B, base_ch, H, W)
        t_emb = self.time_emb(t)

        # encoder
        skips = []
        for enc_block, down in zip(self.encoder, self.downsamples):
            h = enc_block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, t_emb)

        # decoder
        for dec_block, up in zip(self.decoder, self.upsamples):
            h = up(h)  # (B, 2 * C, H / 2, W / 2) -> (B, C, H, W)
            h = torch.cat([h, skips.pop()], dim=1)  # (B, 2 * C, H, W)
            h = dec_block(h, t_emb)  # (B, C, H, W)

        return self.end_conv(h)
