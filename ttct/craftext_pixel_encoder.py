"""
Pixel CNN matching caged_craftext baselines ActorCriticConvWithBERTCMDP (Flax):
three blocks of Conv 5x5 (32) + ReLU + MaxPool 3x3 stride 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CraftextPixelEncoder(nn.Module):
    """Output dim is 288 for Craftax Classic Pixels agent view (63, 63, 3)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().contiguous()

        def _body(t):
            t = F.relu(self.conv1(t))
            t = F.max_pool2d(t, kernel_size=3, stride=3)
            t = F.relu(self.conv2(t))
            t = F.max_pool2d(t, kernel_size=3, stride=3)
            t = F.relu(self.conv3(t))
            t = F.max_pool2d(t, kernel_size=3, stride=3, padding=1)
            return torch.flatten(t, start_dim=1)

        try:
            return _body(x)
        except RuntimeError as e:
            msg = str(e)
            if x.is_cuda and ("cuDNN" in msg or "CUDNN" in msg):
                was = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                try:
                    return _body(x)
                finally:
                    torch.backends.cudnn.enabled = was
            raise

    @staticmethod
    def output_dim(height: int, width: int) -> int:
        with torch.no_grad():
            m = CraftextPixelEncoder()
            y = m(torch.zeros(1, 3, height, width))
            return int(y.shape[1])
