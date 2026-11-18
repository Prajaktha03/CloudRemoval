"""Baseline and ablation architectures for satellite cloud removal experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator import DoubleConv


class OpticalOnlyUNet(nn.Module):
    """Optical-only U-Net ablation without SAR radar input.
    
    Used to scientifically isolate and quantify the empirical performance gain
    provided by multimodal Sentinel-1 SAR fusion.
    """

    def __init__(
        self,
        optical_channels: int = 13,
        use_mask: bool = True,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.optical_channels = optical_channels
        self.use_mask = use_mask
        self.base_channels = base_channels

        in_channels = optical_channels + (1 if use_mask else 0)

        self.inc = DoubleConv(in_channels, base_channels, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8, dropout))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 8, base_channels * 16, dropout))

        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 16, base_channels * 8, dropout)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 8, base_channels * 4, dropout)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_channels * 4, base_channels * 2, dropout)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_channels * 2, base_channels, dropout)

        self.outc = nn.Conv2d(base_channels, optical_channels, kernel_size=1)

    def forward(
        self,
        optical: torch.Tensor,
        cloud_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [optical]
        if self.use_mask:
            if cloud_mask is None:
                cloud_mask = torch.zeros(
                    optical.shape[0], 1, optical.shape[2], optical.shape[3],
                    dtype=optical.dtype, device=optical.device
                )
            parts.append(cloud_mask)

        x = torch.cat(parts, dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5)
        if d1.shape[-2:] != x4.shape[-2:]:
            d1 = F.interpolate(d1, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.conv_up1(torch.cat([x4, d1], dim=1))

        d2 = self.up2(d1)
        if d2.shape[-2:] != x3.shape[-2:]:
            d2 = F.interpolate(d2, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.conv_up2(torch.cat([x3, d2], dim=1))

        d3 = self.up3(d2)
        if d3.shape[-2:] != x2.shape[-2:]:
            d3 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.conv_up3(torch.cat([x2, d3], dim=1))

        d4 = self.up4(d3)
        if d4.shape[-2:] != x1.shape[-2:]:
            d4 = F.interpolate(d4, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.conv_up4(torch.cat([x1, d4], dim=1))

        logits = self.outc(d4)
        return torch.sigmoid(logits)
