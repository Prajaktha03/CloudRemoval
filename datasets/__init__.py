"""Datasets package for multimodal cloud removal in satellite imagery."""

from datasets.alignment import check_geospatial_alignment, validate_array_alignment
from datasets.cloud_masks import compute_cloud_fraction, detect_s2_clouds, detect_cloud_shadows
from datasets.dataset_adapter import MultimodalSampleRecord, SEN12MSCRTSDataset
from datasets.preprocessing import (
    make_rgb_preview,
    make_sar_preview,
    normalize_optical,
    normalize_sar,
    read_geotiff_multimodal,
)

__all__ = [
    "SEN12MSCRTSDataset",
    "MultimodalSampleRecord",
    "normalize_optical",
    "normalize_sar",
    "read_geotiff_multimodal",
    "make_rgb_preview",
    "make_sar_preview",
    "detect_s2_clouds",
    "detect_cloud_shadows",
    "compute_cloud_fraction",
    "check_geospatial_alignment",
    "validate_array_alignment",
]
