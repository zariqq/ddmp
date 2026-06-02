import torch
import torch.nn as nn
from torch import tensor
from utils import TimeEmbedding, SelfAttention, WideResNet
import unittest


class TestUtils(unittest.TestCase):
    def test_te(self):
        with torch.no_grad():
            embed = TimeEmbedding(20, 80)
            t = tensor([5])
            out = embed(t)
            self.assertEqual(out.shape, (1, 80), f"expected (1, 80), got {out.shape}")

            # 2. different timesteps give different embeddings
            t1 = tensor([1])
            t2 = tensor([50])
            self.assertTrue(not torch.allclose(embed(t1), embed(t2)))

            # 3. batch of timesteps works
            t_batch = tensor([1, 10, 50, 100])
            out_batch = embed(t_batch)
            self.assertEqual(out_batch.shape, (4, 80))

            # 4. same t always gives same output (deterministic)
            self.assertTrue(torch.allclose(embed(t1), embed(t1)))

    def test_sa(self):
        x = torch.randn(32, 12, 5, 6)
        sa = SelfAttention(12, 3)

        self.assertEqual(sa(x).shape, x.shape)

        self.assertTrue(torch.allclose(sa(x), x, atol=1e-04))

        x = torch.randn_like(x, requires_grad=True)
        loss = sa(x).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestWideResNet(unittest.TestCase):
    def test_wrn_instantiation(self):
        with torch.no_grad():
            for c in [3, 6, 9]:
                WideResNet(c, 6, 128, 1)
        self.assertTrue(True)

    def test_wrn_out_shape(self):
        with torch.no_grad():
            net = WideResNet(3, 6, 128)

            B, C, H, W = 2, 3, 5, 5
            x = torch.randn(B, C, H, W)
            t = torch.randn(B, 128)
            out = net(x, t)
            expectedShape = (B, 6, H, W)

            self.assertEqual(
                expectedShape,
                out.shape,
                f"expected shape is {expectedShape} but got {out.shape}",
            )

    def test_no_projection_when_channels_match(self):
        model = WideResNet(16, 16, 32)

        x = torch.randn(2, 16, 8, 8)
        t = torch.randn(2, 32)
        out = model(x, t)
        # if no projection, input must have 16 channels already
        self.assertEqual(
            out.shape, (2, 16, 8, 8), f"expected {(2, 16, 8, 8)} but got {out.shape}"
        )
        # check that input_proj is Identity
        self.assertTrue(isinstance(model.proj_res, torch.nn.Identity))

    def test_dropout(self):
        model = WideResNet(
            2, 16, in_num_groups=2, out_num_groups=2, time_emb_dim=32, dropout=0.5
        )
        x = torch.randn(2, 2, 16, 16)
        t = torch.randn(2, 32)

        model.train()
        out1 = model(x, t)
        out2 = model(x, t)
        # with dropout > 0, outputs must differ
        self.assertFalse(
            torch.allclose(out1, out2), "Dropout not stochastic in train mode"
        )

        model.eval()
        with torch.no_grad():
            out3 = model(x, t)
            out4 = model(x, t)
        self.assertTrue(torch.allclose(out3, out4), "Dropout still active in eval mode")

    def test_zero_time_embedding(self):
        model = WideResNet(
            2, 32, in_num_groups=2, out_num_groups=2, time_emb_dim=64, dropout=0.0
        )
        x = torch.randn(2, 2, 32, 32)
        t = torch.zeros(2, 64)

        out = model(x, t)
        self.assertFalse(torch.isnan(out).any(), "NaN in output")
        self.assertFalse(torch.isinf(out).any(), "Inf in output")

    def test_different_time_embeddings_give_different_outputs(self):
        model = WideResNet(
            2, 16, in_num_groups=2, out_num_groups=2, time_emb_dim=32, dropout=0.0
        )
        model.eval()
        x = torch.randn(2, 2, 16, 16)
        t1 = torch.randn(2, 32)
        t2 = torch.randn(2, 32) + 1.0  # different time vectors

        with torch.no_grad():
            out1 = model(x, t1)
            out2 = model(x, t2)
        self.assertFalse(
            torch.allclose(out1, out2), "Time embedding had no effect on output"
        )

    def test_gradient_flow(self):
        model = WideResNet(2, 16, in_num_groups=2, out_num_groups=2, time_emb_dim=32)
        model.train()
        x = torch.randn(2, 2, 16, 16)
        t = torch.randn(2, 32, requires_grad=True)

        out = model(x, t)
        loss = out.sum()
        loss.backward()

        # time embedding must receive gradients
        self.assertIsNotNone(t.grad, "Time input lacks gradient")
        # all trainable parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad is not None, f"Parameter {name} has no gradient"
                )

    def test_global_residual_outside(self):
        """Simulate the global skip connection you mentioned: backbone(x) + proj(x)."""
        in_ch = 2
        out_ch = 32
        backbone = WideResNet(
            in_ch, out_ch, in_num_groups=2, out_num_groups=2, time_emb_dim=64
        )
        x = torch.randn(2, in_ch, 32, 32)
        t = torch.randn(2, 64)

        # Project input to match output channels for global residual
        proj = nn.Conv2d(in_ch, out_ch, 1)
        global_out = backbone(x, t) + proj(x)

        self.assertEqual(global_out.shape, (2, out_ch, 32, 32))


if __name__ == "__main__":
    unittest.main()
