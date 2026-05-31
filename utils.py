import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from math import log as m_log

T = 1000

"""
TODO: implement time embeddings
"""


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


"""
TODO: implement self-attention block
"""


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()


"""
TODO: implement wide ResNet block
"""


def test_te():
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


if __name__ == "__main__":
    test_te()

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
