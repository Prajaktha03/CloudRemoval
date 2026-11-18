"""Conditional PatchGAN Discriminator for Multimodal Satellite Reconstruction."""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN Discriminator.
    
    Distinguishes whether optical image patches are real satellite observations or
    model-generated reconstructions, conditioned on the cloudy optical, SAR, and mask inputs.
    Outputs raw logits for numerically stable BCEWithLogitsLoss.
    """

    def __init__(
        self,
        optical_channels: int = 13,
        sar_channels: int = 2,
        use_mask: bool = True,
        base_channels: int = 32,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.use_mask = use_mask

        # Condition channels (cloudy optical + SAR + optional mask) + candidate optical channels (real or fake)
        in_channels = (optical_channels + sar_channels + (1 if use_mask else 0)) + optical_channels

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend([
                nn.Conv2d(
                    base_channels * nf_mult_prev,
                    base_channels * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(base_channels * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Final patch stride 1 layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv2d(
                base_channels * nf_mult_prev,
                base_channels * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            # 1-channel raw logit prediction patch map
            nn.Conv2d(base_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ])

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        optical_cond: torch.Tensor,
        sar_cond: torch.Tensor,
        candidate_optical: torch.Tensor,
        cloud_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [optical_cond, sar_cond]
        if self.use_mask:
            if cloud_mask is None:
                cloud_mask = torch.zeros(
                    optical_cond.shape[0], 1, optical_cond.shape[2], optical_cond.shape[3],
                    dtype=optical_cond.dtype, device=optical_cond.device
                )
            parts.append(cloud_mask)

        # Concatenate conditioning inputs with candidate image
        parts.append(candidate_optical)
        x = torch.cat(parts, dim=1)
        return self.model(x)
