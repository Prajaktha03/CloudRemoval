"""Preprocessing pipeline for Sentinel-1 SAR and Sentinel-2 optical imagery."""

from pathlib import Path
from typing import Any
import numpy as np
import rasterio


def normalize_optical(
    array: np.ndarray,
    scale_factor: float = 10000.0,
    nodata_value: float | None = None,
) -> np.ndarray:
    """Normalize raw Sentinel-2 DN to surface reflectance in [0, 1].
    
    Args:
        array: Input optical array with shape (H, W, C) or (C, H, W).
        scale_factor: Digital number scaling (10000.0 for ESA S2 products).
        nodata_value: Explicit nodata value to replace with 0.
        
    Returns:
        Normalized array in range [0, 1], dtype float32, with shape (H, W, C).
    """
    if array.ndim == 2:
        array = array[..., np.newaxis]
    elif array.ndim == 3 and array.shape[0] in {1, 2, 3, 4, 12, 13} and array.shape[-1] > 32:
        # Array is in (C, H, W) format, convert to (H, W, C)
        array = np.moveaxis(array, 0, -1)
        
    out = array.astype(np.float32)
    
    if nodata_value is not None:
        out[out == nodata_value] = 0.0
        
    # Replace non-finite pixels
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    
    # If integer values > 1, apply reflectance scale
    if array.max() > 1.0:
        if array.max() <= 255.0 and array.dtype == np.uint8:
            out /= 255.0
        else:
            out /= scale_factor
            
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalize_sar(
    array: np.ndarray,
    min_db: float = -35.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """Normalize Sentinel-1 SAR dB backscatter to [0, 1] preserving radar contrast.
    
    Physical rationale:
    Copernicus Sentinel-1 IW GRD backscatter sigma0 typically spans -35 dB (calm water/smooth terrain)
    to 0 dB (dense urban/corner reflectors). Linear mapping maps this physical dynamic range into [0, 1].
    """
    if array.ndim == 2:
        array = array[..., np.newaxis]
    elif array.ndim == 3 and array.shape[0] in {1, 2, 3} and array.shape[-1] > 32:
        array = np.moveaxis(array, 0, -1)
        
    out = array.astype(np.float32)
    out = np.nan_to_num(out, nan=min_db, posinf=max_db, neginf=min_db)
    
    # If SAR is already in [0, 1], leave it
    if out.min() >= 0.0 and out.max() <= 1.0 and (out.max() - out.min() <= 1.0):
        return out
        
    clipped = np.clip(out, min_db, max_db)
    normalized = (clipped - min_db) / (max_db - min_db)
    return normalized.astype(np.float32)


def read_geotiff_multimodal(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a multi-band GeoTIFF file and return (H, W, C) array and metadata profile."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Satellite GeoTIFF not found: {p}")
        
    with rasterio.open(p) as src:
        data = src.read()  # (C, H, W)
        profile = src.profile.copy()
        
    array = np.moveaxis(data, 0, -1)  # (H, W, C)
    return array, profile


def make_rgb_preview(
    optical: np.ndarray,
    bands: tuple[int, int, int] = (3, 2, 1),  # B4 (Red), B3 (Green), B2 (Blue)
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Generate a clean visual RGB preview from multispectral imagery using 2-98% percentile stretch.
    
    Returns:
        np.ndarray of uint8 [0, 255] with shape (H, W, 3).
    """
    h, w, c = optical.shape
    if c >= 3:
        b_r, b_g, b_b = bands
        # Fallback if band index is out of bounds
        b_r = min(b_r, c - 1)
        b_g = min(b_g, c - 1)
        b_b = min(b_b, c - 1)
        rgb = optical[..., [b_r, b_g, b_b]].astype(np.float32)
    elif c == 1:
        rgb = np.repeat(optical[..., :1], 3, axis=-1).astype(np.float32)
    else:
        rgb = optical[..., :3].astype(np.float32)
        
    # Percentile linear contrast stretch per band
    p_low, p_high = percentile_clip
    out = np.zeros_like(rgb)
    for b in range(3):
        low = np.percentile(rgb[..., b], p_low)
        high = np.percentile(rgb[..., b], p_high)
        if high > low:
            out[..., b] = (rgb[..., b] - low) / (high - low)
        else:
            out[..., b] = rgb[..., b]
            
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def make_sar_preview(
    sar: np.ndarray,
    polarization_indices: tuple[int, int] = (0, 1),  # VV, VH
) -> np.ndarray:
    """Generate an RGB composite visualization of SAR backscatter (VV, VH, VV/VH ratio)."""
    h, w, c = sar.shape
    if c >= 2:
        vv = sar[..., polarization_indices[0]]
        vh = sar[..., polarization_indices[1]]
        ratio = np.clip(vv - vh, 0.0, 1.0)
        # Create dual-pol false color composite: R=VV, G=VH, B=Ratio
        composite = np.stack([vv, vh, ratio], axis=-1)
    else:
        # Single pol: grayscale
        composite = np.repeat(sar[..., :1], 3, axis=-1)
        
    return np.clip(composite * 255.0, 0, 255).astype(np.uint8)
