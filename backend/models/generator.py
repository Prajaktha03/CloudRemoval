import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class MultimodalUNet(nn.Module):
    """U-Net generator for normalized optical, SAR, and optional cloud-mask inputs."""

    def __init__(
        self,
        optical_channels: int = 13,
        sar_channels: int = 2,
        use_mask: bool = True,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        input_channels = optical_channels + sar_channels + int(use_mask)
        self.use_mask = use_mask
        self.encoder1 = ConvBlock(input_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.decoder4 = ConvBlock(base_channels * 24, base_channels * 8)
        self.decoder3 = ConvBlock(base_channels * 12, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, optical_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    @staticmethod
    def _upsample(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        cloud_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [optical, sar]
        if self.use_mask:
            if cloud_mask is None:
                cloud_mask = optical.new_zeros((optical.shape[0], 1, *optical.shape[-2:]))
            parts.append(cloud_mask)
        x = torch.cat(parts, dim=1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(torch.cat([self._upsample(bottleneck, e4), e4], dim=1))
        d3 = self.decoder3(torch.cat([self._upsample(d4, e3), e3], dim=1))
        d2 = self.decoder2(torch.cat([self._upsample(d3, e2), e2], dim=1))
        d1 = self.decoder1(torch.cat([self._upsample(d2, e1), e1], dim=1))
        return torch.sigmoid(self.output(d1))
