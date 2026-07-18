from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """Three-layer SRCNN for one luminance channel.

    Training uses valid convolutions, as in the Caffe training network.  
    During full-image testing, the official MATLAB code uses replicate-padded 
    ``same`` filtering.  ``forward_same`` reproduces that test behaviour.
    """

    def __init__(
        self,
        n1: int = 64,
        n2: int = 32,
        f1: int = 9,
        f2: int = 1,
        f3: int = 5,
    ) -> None:
        super().__init__()
        for name, value in {"f1": f1, "f2": f2, "f3": f3}.items():
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}")

        self.n1, self.n2 = int(n1), int(n2)
        self.f1, self.f2, self.f3 = int(f1), int(f2), int(f3)

        self.conv1 = nn.Conv2d(1, self.n1, self.f1, padding=0)
        self.conv2 = nn.Conv2d(self.n1, self.n2, self.f2, padding=0)
        self.conv3 = nn.Conv2d(self.n2, 1, self.f3, padding=0)
        self.reset_parameters_original()

    @property
    def output_shrinkage(self) -> int:
        return (self.f1 - 1) + (self.f2 - 1) + (self.f3 - 1)

    @property
    def training_border(self) -> int:
        return self.output_shrinkage // 2

    def reset_parameters_original(self) -> None:
        # Original Caffe network: Gaussian(0, 0.001) weights, zero biases.
        for layer in (self.conv1, self.conv2, self.conv3):
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return self.conv3(x)

    @staticmethod
    def _replicate_same_conv(x: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
        k_h, k_w = layer.kernel_size
        pad_h, pad_w = k_h // 2, k_w // 2
        if pad_h or pad_w:
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        return F.conv2d(x, layer.weight, layer.bias, stride=1, padding=0)

    def forward_same(self, x: torch.Tensor) -> torch.Tensor:
        """Official test-style same-size inference with replicate padding."""
        x = F.relu(self._replicate_same_conv(x, self.conv1), inplace=True)
        x = F.relu(self._replicate_same_conv(x, self.conv2), inplace=True)
        return self._replicate_same_conv(x, self.conv3)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
