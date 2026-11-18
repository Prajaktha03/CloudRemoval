"""Loss functions for Multimodal Satellite Cloud Removal."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudAwareReconstructionLoss(nn.Module):
    """Mask-weighted L1 reconstruction loss focusing heavily on cloud-obscured regions.
    
    L_recon = lambda_cloud * L1(cloud_region) + lambda_clear * L1(clear_region)
    """

    def __init__(
        self,
        lambda_cloud: float = 3.0,
        lambda_clear: float = 1.0,
        use_mask_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_cloud = lambda_cloud
        self.lambda_clear = lambda_clear
        self.use_mask_weighting = use_mask_weighting

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        diff = torch.abs(reconstruction - target)

        if not self.use_mask_weighting or cloud_mask is None:
            l1_mean = torch.mean(diff)
            return l1_mean, {"l1_total": l1_mean.item(), "l1_cloud": l1_mean.item(), "l1_clear": l1_mean.item()}

        # Cloud mask shape: (B, 1, H, W) -> broadcast across spectral channels
        mask = (cloud_mask > 0.5).float()
        inv_mask = 1.0 - mask

        cloud_pixels = torch.sum(mask) * target.shape[1]
        clear_pixels = torch.sum(inv_mask) * target.shape[1]

        # Cloud region L1
        if cloud_pixels > 0:
            l1_cloud = torch.sum(diff * mask) / cloud_pixels
        else:
            l1_cloud = torch.tensor(0.0, device=diff.device)

        # Clear region L1
        if clear_pixels > 0:
            l1_clear = torch.sum(diff * inv_mask) / clear_pixels
        else:
            l1_clear = torch.tensor(0.0, device=diff.device)

        total_loss = self.lambda_cloud * l1_cloud + self.lambda_clear * l1_clear

        return total_loss, {
            "l1_total": total_loss.item(),
            "l1_cloud": l1_cloud.item(),
            "l1_clear": l1_clear.item(),
        }


class GANLoss(nn.Module):
    """Adversarial loss using numerically stable BCEWithLogits."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            return torch.ones_like(prediction)
        return torch.zeros_like(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


def composite_cloud_free(
    reconstruction: torch.Tensor | np.ndarray,
    optical_input: torch.Tensor | np.ndarray,
    cloud_mask: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Mask-aware compositing: preserves clear pixels from original observation and replaces cloudy regions.
    
    Y_comp = M * Y_recon + (1 - M) * X_opt
    """
    if isinstance(reconstruction, torch.Tensor):
        mask = (cloud_mask > 0.5).float()
        return mask * reconstruction + (1.0 - mask) * optical_input
    else:
        mask = (cloud_mask > 0.5).astype(reconstruction.dtype)
        return mask * reconstruction + (1.0 - mask) * optical_input
