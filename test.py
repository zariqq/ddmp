import torch
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


if __name__ == "__main__":
    unittest.main()
