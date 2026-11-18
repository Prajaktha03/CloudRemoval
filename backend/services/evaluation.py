import numpy as np
from skimage.metrics import structural_similarity

from backend.core.errors import InputValidationError


def evaluate(reconstruction: np.ndarray, reference: np.ndarray) -> dict[str, float | bool | None]:
    if reconstruction.shape != reference.shape:
        raise InputValidationError("Reconstruction and reference must have identical dimensions and channels")
    if reconstruction.ndim != 3 or not np.isfinite(reconstruction).all() or not np.isfinite(reference).all():
        raise InputValidationError("Evaluation images must be finite HWC arrays")
    if reconstruction.min() < 0 or reconstruction.max() > 1 or reference.min() < 0 or reference.max() > 1:
        raise InputValidationError("Evaluation image values must be normalized to [0, 1]")
    if min(reconstruction.shape[:2]) < 7:
        raise InputValidationError("Evaluation images must be at least 7 by 7 pixels")
    mean_squared_error = float(np.mean(np.square(reference - reconstruction)))
    mean_absolute_error = float(np.mean(np.abs(reference - reconstruction)))
    psnr = float("inf") if mean_squared_error == 0 else float(10 * np.log10(1.0 / mean_squared_error))
    ssim_reference = reference[..., 0] if reference.shape[2] == 1 else reference
    ssim_reconstruction = reconstruction[..., 0] if reconstruction.shape[2] == 1 else reconstruction
    return {
        "psnr": psnr if np.isfinite(psnr) else None,
        "psnr_is_infinite": bool(np.isinf(psnr)),
        "mae": mean_absolute_error,
        "rmse": float(np.sqrt(mean_squared_error)),
        "ssim": float(
            structural_similarity(
                ssim_reference,
                ssim_reconstruction,
                data_range=1.0,
                channel_axis=-1 if reconstruction.shape[2] > 1 else None,
            )
        ),
    }
