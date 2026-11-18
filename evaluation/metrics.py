"""Quantitative evaluation metrics for remote-sensing image reconstruction.

Calculates:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Region-specific evaluations: Full image, Cloud regions (M=1), Clear regions (M=0).
"""

from typing import Any
import numpy as np
from skimage.metrics import structural_similarity


def calculate_psnr(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    data_range: float = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    if mask is not None:
        m = (mask > 0.5).astype(bool)
        if m.ndim == 2:
            m = m[..., np.newaxis]
        if not np.any(m):
            return float("nan")
        diff = (reconstruction - reference)[np.broadcast_to(m, reference.shape)]
        mse = float(np.mean(np.square(diff)))
    else:
        mse = float(np.mean(np.square(reconstruction - reference)))

    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def calculate_ssim(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Calculate multispectral or single-band Structural Similarity Index (SSIM)."""
    h, w = reference.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return float("nan")

    c = reference.shape[-1] if reference.ndim == 3 else 1
    if c == 1 or reference.ndim == 2:
        ref_2d = reference[..., 0] if reference.ndim == 3 else reference
        rec_2d = reconstruction[..., 0] if reconstruction.ndim == 3 else reconstruction
        return float(structural_similarity(ref_2d, rec_2d, data_range=data_range, win_size=win_size))

    # Multispectral average
    ssim_vals = []
    for band in range(c):
        val = structural_similarity(
            reference[..., band],
            reconstruction[..., band],
            data_range=data_range,
            win_size=win_size,
        )
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))


def calculate_mae(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    if mask is not None:
        m = (mask > 0.5).astype(bool)
        if m.ndim == 2:
            m = m[..., np.newaxis]
        if not np.any(m):
            return float("nan")
        diff = np.abs(reconstruction - reference)[np.broadcast_to(m, reference.shape)]
        return float(np.mean(diff))
    return float(np.mean(np.abs(reconstruction - reference)))


def calculate_rmse(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    if mask is not None:
        m = (mask > 0.5).astype(bool)
        if m.ndim == 2:
            m = m[..., np.newaxis]
        if not np.any(m):
            return float("nan")
        diff = (reconstruction - reference)[np.broadcast_to(m, reference.shape)]
        return float(np.sqrt(np.mean(np.square(diff))))
    return float(np.sqrt(np.mean(np.square(reconstruction - reference))))


def evaluate_sample_metrics(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    cloud_mask: np.ndarray | None = None,
    data_range: float = 1.0,
) -> dict[str, Any]:
    """Compute complete scientific metric suite for a sample pair."""
    # Ensure HWC
    if reconstruction.ndim == 2:
        reconstruction = reconstruction[..., np.newaxis]
    if reference.ndim == 2:
        reference = reference[..., np.newaxis]

    metrics = {
        "psnr_full": calculate_psnr(reconstruction, reference, data_range),
        "ssim_full": calculate_ssim(reconstruction, reference, data_range),
        "mae_full": calculate_mae(reconstruction, reference),
        "rmse_full": calculate_rmse(reconstruction, reference),
    }

    if cloud_mask is not None:
        c_mask = cloud_mask > 0.5
        inv_mask = ~c_mask
        has_clouds = np.any(c_mask)
        has_clear = np.any(inv_mask)

        metrics["psnr_cloud"] = calculate_psnr(reconstruction, reference, data_range, c_mask) if has_clouds else None
        metrics["mae_cloud"] = calculate_mae(reconstruction, reference, c_mask) if has_clouds else None
        metrics["rmse_cloud"] = calculate_rmse(reconstruction, reference, c_mask) if has_clouds else None

        metrics["psnr_clear"] = calculate_psnr(reconstruction, reference, data_range, inv_mask) if has_clear else None
        metrics["mae_clear"] = calculate_mae(reconstruction, reference, inv_mask) if has_clear else None
        metrics["rmse_clear"] = calculate_rmse(reconstruction, reference, inv_mask) if has_clear else None

    return metrics
