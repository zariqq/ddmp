import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch import tensor
from math import log as m_log

from config import T


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
        self, in_channels, out_channels, time_emb_dim, num_groups, dropout=0.1
    ):
        super().__init__()

        norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.proj_res = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=1
        )

        self.backbone_first = nn.Sequential(
            norm,  # (B, Cin, H, W)
            nn.SiLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),  # (B, Cout, H, W)
        )
        self.backbone_second = nn.Sequencial(
            norm,
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, time_emb):
        residual = self.proj_res(x)
        bb1 = self.backbone_first(x)  # (B, Cout, H, W)
        return self.backbone_second(bb1 + self.time_proj(time_emb)) + residual


"""
TODO: implement U-Net (the whole model with Wide ResNet and Attention blocks)
"""


def test_te():
    with torch.no_grad():
        embed = TimeEmbedding(20, 80)
        t = tensor([5])
        out = embed(t)
        assert out.shape == (1, 80), f"expected (1, 80), got {out.shape}"

        # 2. different timesteps give different embeddings
        t1 = tensor([1])
        t2 = tensor([50])
        assert not torch.allclose(embed(t1), embed(t2))

        # 3. batch of timesteps works
        t_batch = tensor([1, 10, 50, 100])
        out_batch = embed(t_batch)
        assert out_batch.shape == (4, 80)

        # 4. same t always gives same output (deterministic)
        assert torch.allclose(embed(t1), embed(t1))


def test_sa():
    x = torch.randn(32, 12, 5, 6)
    sa = SelfAttention(12, 3)

    assert sa(x).shape == x.shape

    assert torch.allclose(sa(x), x, atol=1e-04)

    x = torch.randn_like(x, requires_grad=True)
    loss = sa(x).sum()
    loss.backward()
    assert x.grad is not None


if __name__ == "__main__":
    test_te()
    test_sa()

    # import matplotlib.pyplot as plt

    # plt.imshow(
    #     TimeEmbedding.positional_encodings(1000, 250),
    #     aspect="auto",
    #     cmap="RdBu",
    #     vmin=-1.0,
    #     vmax=1.0,
    # )

    # plt.tight_layout()
    # plt.show()
