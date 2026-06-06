import unittest

import torch
import torch.nn as nn
from torch import tensor

from utils import SelfAttention, TimeEmbedding, WideResNet, UNet, Hypers


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

            # 3. B of timesteps works
            t_B = tensor([1, 10, 50, 100])
            out_B = embed(t_B)
            self.assertEqual(out_B.shape, (4, 80))

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


B, C, H, W = 4, 3, 32, 32
hyp = Hypers(128)
model = UNet(hyp)


class TestUNet(unittest.TestCase):

    # ---------- Test 1: Basic forward shape ----------
    def test_output_shape(self):
        x = torch.randn(B, C, H, W)
        t = torch.randint(0, 1000, (B,))
        out = model(x, t)
        self.assertEqual(
            out.shape, x.shape, f"Expected shape {x.shape}, got {out.shape}"
        )

    # ---------- Test 2: Different B sizes ----------
    def test_B_sizes(self):
        for bs in [1, 2, 8]:
            x = torch.randn(bs, C, H, W)
            t = torch.randint(0, 1000, (bs,))
            out = model(x, t)
            self.assertEqual(out.shape, (bs, C, H, W), f"B {bs} failed")

    # ---------- Test 3: Different spatial sizes (must be divisible by 16) ----------
    def test_spatial_sizes(self):
        for size in [32, 64, 128]:
            x = torch.randn(2, C, size, size)
            t = torch.randint(0, 1000, (2,))
            out = model(x, t)
            self.assertEqual(
                out.shape, (2, C, size, size), f"Spatial size {size} failed"
            )

    # ---------- Test 4: Gradient flow ----------
    def test_gradients(self):
        model.train()
        x = torch.randn(B, C, H, W, requires_grad=False)
        t = torch.randint(0, 1000, (B,))
        out = model(x, t)
        loss = out.mean()
        loss.backward()
        # Check that at least some parameters received gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        self.assertTrue(has_grad, "No gradients flowing")

    # ---------- Test 5: Time embedding works with different t shapes ----------
    def test_time_inputs(self):
        # Single integer vs 1D tensor
        model.eval()
        x = torch.randn(1, C, H, W)
        t_scalar = 42
        out1 = model(x, torch.tensor([t_scalar]))
        out2 = model(x, torch.tensor([t_scalar]))
        self.assertTrue(
            torch.allclose(out1, out2), "Repeated run with same t should be identical"
        )

        # Bed t
        t_B = torch.randint(0, 1000, (4,))
        out = model(torch.randn(4, C, H, W), t_B)
        self.assertEqual(out.shape[0], 4, "Bed time failed")

    # ---------- Test 6: Output not constant ----------
    def test_output_variation(self):
        model.eval()
        x = torch.randn(4, C, H, W)
        t = torch.randint(0, 1000, (4,))
        out = model(x, t)
        # Check that output varies across B
        self.assertFalse(torch.allclose(out[0], out[1]), "Output should vary across B")

    # ---------- Test 7: Quick training step (overfitting on a single B) ----------
    def test_training_step(self):
        model.train()
        x = torch.randn(8, C, H, W)
        t = torch.randint(0, 1000, (8,))
        noise = torch.randn_like(x)
        # Simple MSE loss like DDPM training (predict noise)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(5):
            optim.zero_grad()
            pred = model(x, t)
            loss = ((pred - noise) ** 2).mean()
            loss.backward()
            optim.step()
        # Loss should decrease if model can learn something
        final_loss = loss.item()
        self.assertTrue(final_loss < 1.0, f"Loss too high after 5 steps: {final_loss}")


if __name__ == "__main__":
    unittest.main()
