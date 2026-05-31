"""
TODO:
- make for batch
- image (x0) is normalized to value range [-1, 1]
- implement model (U-Net) that predicts NOISE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor

from config import T

noise_schedule = torch.linspace(10**-4, 0.02, T)
noise_std = torch.sqrt(noise_schedule)
alpha = 1 - noise_schedule
alpha_prod = torch.cumprod(alpha, 0)

image_shape = (3, 16, 16)

model = nn.Module()


def train(images, epoch=10):
    model.train()
    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters())

    for x0 in images:
        t = torch.randint(1, T + 1)
        eps = torch.randn_like(x0)

        optimizer.zero_grad()

        xt = torch.sqrt(alpha_prod[t]) * x0 + torch.sqrt(1 - alpha_prod[t]) * eps
        pred_noise = model(xt, t)

        loss = criteria(pred_noise, eps)
        loss.backward()
        optimizer.step()


def sample():
    xt = torch.randn(image_shape)  # start from pure noise
    for t in range(T, 0):
        z = torch.randn_like(xt) if t > 0 else 0
        pred_noise = model(xt, t)
        xt = (
            xt - (1 - alpha[t]) / torch.sqrt(1 - alpha_prod[t]) * pred_noise
        ) / torch.sqrt(alpha[t]) + noise_std[t] * z
    return xt
