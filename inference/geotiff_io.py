"""Geospatial I/O utility for GeoTIFF and satellite image formats."""

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile


def read_satellite_image(
    file_bytes_or_path: bytes | str | Path,
    filename: str = "image.tif",
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Read a satellite image (GeoTIFF, PNG, JPEG) and return (H, W, C) array and profile."""
    suffix = Path(filename).suffix.lower()
    
    if suffix in {".tif", ".tiff"}:
        if isinstance(file_bytes_or_path, (str, Path)):
            with rasterio.open(file_bytes_or_path) as src:
                data = src.read()
                profile = src.profile.copy()
        else:
            with MemoryFile(file_bytes_or_path) as mem:
                with mem.open() as src:
                    data = src.read()
                    profile = src.profile.copy()
        array = np.moveaxis(data, 0, -1)
        return array, profile

    # Image file (PNG, JPEG)
    if isinstance(file_bytes_or_path, (str, Path)):
        img = Image.open(file_bytes_or_path)
    else:
        img = Image.open(io.BytesIO(file_bytes_or_path))
        
    array = np.asarray(img)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    return array, None


def write_satellite_image(
    array: np.ndarray,
    output_path: str | Path,
    profile: dict[str, Any] | None = None,
) -> Path:
    """Write reconstructed optical imagery preserving geospatial profile if present."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    h, w, c = array.shape

    if profile is not None and out_p.suffix.lower() in {".tif", ".tiff"}:
        prof = profile.copy()
        prof.update(
            count=c,
            height=h,
            width=w,
            dtype="float32",
        )
        with rasterio.open(out_p, "w", **prof) as dst:
            dst.write(np.moveaxis(array.astype(np.float32), -1, 0))
        return out_p

    # Standard PNG output
    if c in {1, 3, 4}:
        clipped = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        if c == 1:
            clipped = clipped[..., 0]
        img = Image.fromarray(clipped)
        img.save(out_p)
        return out_p

    # Multispectral numpy format
    npz_path = out_p.with_suffix(".npz")
    np.savez_compressed(npz_path, data=array.astype(np.float32))
    return npz_path
