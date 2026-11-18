"""Simple reconstruction baselines for benchmark comparisons."""

import cv2
import numpy as np


def passthrough_baseline(cloudy_optical: np.ndarray) -> np.ndarray:
    """Pass-through baseline: uses raw cloudy observation directly as output.
    
    Serves as an experimental control demonstrating degraded metrics caused by cloud contamination.
    """
    return cloudy_optical.copy()


def spatial_inpaint_baseline(cloudy_optical: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """Bicubic / Navier-Stokes spatial inpainting baseline on cloud-covered regions.
    
    Reconstructs obscured pixels from adjacent clear pixels using spatial interpolation.
    """
    h, w, c = cloudy_optical.shape
    mask_2d = (cloud_mask[..., 0] > 0.5).astype(np.uint8) * 255
    
    if not np.any(mask_2d):
        return cloudy_optical.copy()
        
    out = np.zeros_like(cloudy_optical)
    # Process band by band
    for b in range(c):
        band_u8 = (np.clip(cloudy_optical[..., b], 0.0, 1.0) * 255.0).astype(np.uint8)
        inpainted_u8 = cv2.inpaint(band_u8, mask_2d, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        out[..., b] = inpainted_u8.astype(np.float32) / 255.0
        
    return out
