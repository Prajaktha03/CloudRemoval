"""Cloud and cloud-shadow detection and mask generation for optical satellite imagery."""

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation


def detect_s2_clouds(
    optical: np.ndarray,
    blue_threshold: float = 0.20,
    cirrus_threshold: float = 0.015,
    whiteness_threshold: float = 0.25,
    dilation_radius: int = 3,
) -> np.ndarray:
    """Detect cloud-obscured pixels from Sentinel-2 multispectral or RGB imagery.
    
    Args:
        optical: Normalized reflectance array in range [0, 1] with shape (H, W, C).
        blue_threshold: Reflectance threshold for blue band (B2 or B0).
        cirrus_threshold: Threshold for Sentinel-2 cirrus band (B10).
        whiteness_threshold: Tolerance for RGB spectral flatness.
        dilation_radius: Morphological expansion radius to cover cloud boundaries.
        
    Returns:
        Binary mask array of shape (H, W, 1) where 1.0 = obscured and 0.0 = clear.
    """
    h, w, c = optical.shape
    
    if c >= 13:
        # Sentinel-2 full 13-band L1C/L2A layout
        # B1: 0 (Coastal), B2: 1 (Blue), B3: 2 (Green), B4: 3 (Red)
        # B8: 7 (NIR), B10: 10 (Cirrus), B11: 11 (SWIR1)
        blue = optical[..., 1]
        green = optical[..., 2]
        red = optical[..., 3]
        nir = optical[..., 7]
        cirrus = optical[..., 10]
        swir1 = optical[..., 11]
        
        # 1. Opaque cloud criterion: High blue and visible brightness
        mean_vis = (red + green + blue) / 3.0
        # Whiteness test: difference from mean visible reflectance
        whiteness = (
            np.abs(red - mean_vis) + np.abs(green - mean_vis) + np.abs(blue - mean_vis)
        ) / (mean_vis + 1e-6)
        
        # Opaque clouds: bright in blue, relatively white, and bright in NIR
        cloud_opaque = (blue > blue_threshold) & (mean_vis > 0.18) & (whiteness < whiteness_threshold)
        
        # 2. Cirrus clouds: High cirrus band signal (1.375 um)
        cloud_cirrus = (
            (cirrus > cirrus_threshold)
            & (mean_vis > 0.12)
            & (blue > 0.10)
        )
        
        # 3. Water body rejection: Water has high blue but near-zero NIR/SWIR
        ndwi = (green - nir) / (green + nir + 1e-6)
        is_water = (ndwi > 0.1) & (nir < 0.15) & (blue < 0.35)
        
        # Snow / ice rejection using NDSI = (Green - SWIR1) / (Green + SWIR1)
        ndsi = (green - swir1) / (green + swir1 + 1e-6)
        is_snow = (ndsi > 0.42) & (nir > 0.25)
        
        raw_mask = (cloud_opaque | cloud_cirrus) & (~is_water) & (~is_snow)
        
    elif c >= 3:
        # RGB optical image
        # Assumes bands 0=Red, 1=Green, 2=Blue (or 0=B4, 1=B3, 2=B2)
        r, g, b = optical[..., 0], optical[..., 1], optical[..., 2]
        mean_rgb = (r + g + b) / 3.0
        diff = np.maximum.reduce([np.abs(r - mean_rgb), np.abs(g - mean_rgb), np.abs(b - mean_rgb)])
        saturation = diff / (mean_rgb + 1e-6)
        
        # Clouds are bright and non-saturated (white/gray)
        raw_mask = (mean_rgb > 0.28) & (saturation < 0.30)
    else:
        # Single band: threshold on brightness
        raw_mask = optical[..., 0] > 0.35
        
    # Morphological processing: close small holes and expand cloud edges
    if dilation_radius > 0:
        struct = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=bool)
        processed = binary_closing(raw_mask, structure=struct)
        processed = binary_dilation(processed, structure=struct)
    else:
        processed = raw_mask
        
    return processed.astype(np.float32)[..., np.newaxis]


def detect_cloud_shadows(
    optical: np.ndarray,
    cloud_mask: np.ndarray,
    nir_shadow_threshold: float = 0.12,
) -> np.ndarray:
    """Identify potential cloud shadow regions in multispectral imagery."""
    h, w, c = optical.shape
    if c >= 8:
        nir = optical[..., 7]
        # Shadows are dark in NIR but not under cloud itself
        is_shadow = (nir < nir_shadow_threshold) & (cloud_mask[..., 0] < 0.5)
        struct = np.ones((3, 3), dtype=bool)
        shadow_mask = binary_closing(is_shadow, structure=struct)
        return shadow_mask.astype(np.float32)[..., np.newaxis]
    return np.zeros((h, w, 1), dtype=np.float32)


def compute_cloud_fraction(mask: np.ndarray) -> float:
    """Compute the fraction of cloud-obscured pixels (0.0 to 1.0)."""
    return float(np.mean(mask > 0.5))
