"""Geospatial alignment and correspondence validation between Sentinel-1 and Sentinel-2."""

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class AlignmentReport:
    is_aligned: bool
    shape_match: bool
    crs_match: bool
    transform_match: bool
    bounds_match: bool
    details: dict[str, Any]


def validate_array_alignment(
    optical_shape: tuple[int, ...],
    sar_shape: tuple[int, ...],
    mask_shape: tuple[int, ...] | None = None,
) -> bool:
    """Validate spatial dimensions (H, W) match across modalities."""
    opt_hw = optical_shape[:2]
    sar_hw = sar_shape[:2]
    if opt_hw != sar_hw:
        return False
    if mask_shape is not None and mask_shape[:2] != opt_hw:
        return False
    return True


def check_geospatial_alignment(
    optical_profile: dict[str, Any] | None,
    sar_profile: dict[str, Any] | None,
    tolerance: float = 1e-6,
) -> AlignmentReport:
    """Verify coordinate reference systems, geotransforms, and bounding footprints."""
    details: dict[str, Any] = {}
    
    if optical_profile is None or sar_profile is None:
        return AlignmentReport(
            is_aligned=False,
            shape_match=False,
            crs_match=False,
            transform_match=False,
            bounds_match=False,
            details={"note": "One or both profiles lack geospatial metadata (non-GeoTIFF format)."},
        )
    
    # Check dimensions
    opt_h, opt_w = optical_profile.get("height", 0), optical_profile.get("width", 0)
    sar_h, sar_w = sar_profile.get("height", 0), sar_profile.get("width", 0)
    shape_match = (opt_h == sar_h) and (opt_w == sar_w)
    details["optical_dims"] = (opt_h, opt_w)
    details["sar_dims"] = (sar_h, sar_w)
    
    # Check CRS
    opt_crs = str(optical_profile.get("crs", "")).upper()
    sar_crs = str(sar_profile.get("crs", "")).upper()
    crs_match = bool(opt_crs and sar_crs and opt_crs == sar_crs)
    details["optical_crs"] = opt_crs
    details["sar_crs"] = sar_crs
    
    # Check Affine Transform
    opt_transform = optical_profile.get("transform")
    sar_transform = sar_profile.get("transform")
    transform_match = False
    if opt_transform is not None and sar_transform is not None:
        try:
            t1 = np.array(opt_transform[:6], dtype=np.float64)
            t2 = np.array(sar_transform[:6], dtype=np.float64)
            transform_match = bool(np.allclose(t1, t2, atol=tolerance, rtol=0))
        except (ValueError, TypeError):
            transform_match = False
            
    details["transform_match"] = transform_match
    bounds_match = shape_match and crs_match and transform_match
    is_aligned = shape_match and crs_match and transform_match
    
    return AlignmentReport(
        is_aligned=is_aligned,
        shape_match=shape_match,
        crs_match=crs_match,
        transform_match=transform_match,
        bounds_match=bounds_match,
        details=details,
    )
