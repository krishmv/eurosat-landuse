"""Models compared in this study.

- SmallCNN: a compact network trained from scratch, as a reference point for
  what a model with no prior knowledge reaches on 64x64 tiles.
- ResNet-18, from scratch or initialised with ImageNet weights.
- ResNet-18 with 13 input channels. ImageNet weights only exist for 3 channels,
  so the first convolution is rebuilt (see `expand_first_conv`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    """Four conv blocks (64 -> 32 -> 16 -> 8 -> 4 pixels), then global average pooling."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 32):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(block(in_channels, w), block(w, 2 * w),
                                      block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(8 * w, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def expand_first_conv(conv: nn.Conv2d, in_channels: int, rgb_positions: list[int]) -> nn.Conv2d:
    """Turn a pretrained 3-channel convolution into an `in_channels` one.

    - The red, green and blue filters are copied to the positions of bands
      B04, B03 and B02, so the RGB part of the input is processed exactly as
      the pretrained network expects.
    - Every other band starts from the mean of the three RGB filters, a neutral
      guess that fine-tuning then adapts.
    - All weights are scaled by 3 / in_channels. A conv output is a sum over
      input channels, so without this the first layer's activations would be
      roughly in_channels / 3 times larger than what the rest of the pretrained
      network was trained on.
    """
    w = conv.weight.data                                   # (64, 3, 7, 7)
    new_w = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
    for src, dst in enumerate(rgb_positions):
        new_w[:, dst] = w[:, src]
    new_w *= 3.0 / in_channels

    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    new.weight.data.copy_(new_w)
    return new


def build_model(arch: str, in_channels: int, num_classes: int, pretrained: bool,
                rgb_positions: list[int] | None = None) -> nn.Module:
    if arch == "smallcnn":
        if pretrained:
            raise ValueError("SmallCNN has no pretrained weights")
        return SmallCNN(in_channels, num_classes)

    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels != 3:
            if pretrained:
                model.conv1 = expand_first_conv(model.conv1, in_channels, rgb_positions)
            else:
                model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
