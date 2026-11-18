"""Dataset inspection utility for SEN12MS-CR-TS and multimodal satellite data.
Produces a scientific metadata and integrity report.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from datasets.alignment import check_geospatial_alignment
from datasets.cloud_masks import detect_s2_clouds, compute_cloud_fraction
from datasets.preprocessing import normalize_optical, normalize_sar


def inspect_dataset(data_dir: str | Path, output_json: str | Path | None = None) -> dict[str, Any]:
    root = Path(data_dir)
    print(f"================================================================================")
    print(f"SEN12MS-CR-TS Multimodal Dataset Inspection Report")
    print(f"Directory: {root.resolve()}")
    print(f"================================================================================")
    
    if not root.is_dir():
        report = {
            "status": "error",
            "message": f"Dataset directory not found: {root}",
            "file_count": 0,
        }
        print(f"ERROR: {report['message']}")
        return report

    all_tifs = glob.glob(str(root / "**" / "*.tif"), recursive=True)
    pattern = re.compile(r"(s1|s2)_ROIs(\d+)_(\d+)_ImgNo_(\d+)_([\d-]+)_patch_(\d+)\.tif")
    
    s1_files = []
    s2_files = []
    other_files = []
    corrupted_files = []
    rois = set()
    patches = set()
    dates = set()

    for p in all_tifs:
        fname = os.path.basename(p)
        m = pattern.match(fname)
        if m:
            sensor, year, roi_id, img_no, date, patch_id = m.groups()
            roi_key = f"ROI_{year}_{roi_id}"
            rois.add(roi_key)
            patches.add(f"{roi_key}_{patch_id}")
            dates.add(date)
            if sensor == "s1":
                s1_files.append(p)
            else:
                s2_files.append(p)
        else:
            other_files.append(p)

    print(f"Total GeoTIFF (.tif) files found: {len(all_tifs)}")
    print(f"Sentinel-1 (SAR) files:           {len(s1_files)}")
    print(f"Sentinel-2 (Optical) files:       {len(s2_files)}")
    print(f"Unique ROIs:                      {len(rois)} ({sorted(rois)})")
    print(f"Unique Geographic Patches:        {len(patches)}")
    print(f"Acquisition Dates range:          {min(dates) if dates else 'N/A'} to {max(dates) if dates else 'N/A'}")
    print(f"--------------------------------------------------------------------------------")

    sample_s1_meta = {}
    sample_s2_meta = {}
    alignment_report = {}

    # Inspect sample Sentinel-1 file
    if s1_files:
        sample_s1 = s1_files[0]
        try:
            with rasterio.open(sample_s1) as src:
                arr_s1 = src.read()
                norm_s1 = normalize_sar(np.moveaxis(arr_s1, 0, -1))
                sample_s1_meta = {
                    "file": os.path.basename(sample_s1),
                    "channels": src.count,
                    "height": src.height,
                    "width": src.width,
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs),
                    "bounds": list(src.bounds),
                    "transform": list(src.transform)[:6],
                    "raw_min": float(arr_s1.min()),
                    "raw_max": float(arr_s1.max()),
                    "raw_mean": float(arr_s1.mean()),
                    "norm_min": float(norm_s1.min()),
                    "norm_max": float(norm_s1.max()),
                }
                print(f"Sample Sentinel-1 SAR Metadata:")
                print(f"  Filename:   {sample_s1_meta['file']}")
                print(f"  Dimensions: {sample_s1_meta['channels']} bands x {sample_s1_meta['height']} x {sample_s1_meta['width']}")
                print(f"  Raw Dtype:  {sample_s1_meta['dtype']} (dB backscatter)")
                print(f"  CRS:        {sample_s1_meta['crs']}")
                print(f"  Raw Range:  [{sample_s1_meta['raw_min']:.2f}, {sample_s1_meta['raw_max']:.2f}] dB (mean: {sample_s1_meta['raw_mean']:.2f} dB)")
                print(f"  Normalized: [{sample_s1_meta['norm_min']:.3f}, {sample_s1_meta['norm_max']:.3f}]")
        except Exception as e:
            corrupted_files.append((sample_s1, str(e)))

    print(f"--------------------------------------------------------------------------------")
    # Inspect sample Sentinel-2 file
    if s2_files:
        sample_s2 = s2_files[0]
        try:
            with rasterio.open(sample_s2) as src:
                arr_s2 = src.read()
                norm_s2 = normalize_optical(np.moveaxis(arr_s2, 0, -1))
                mask = detect_s2_clouds(norm_s2)
                sample_s2_meta = {
                    "file": os.path.basename(sample_s2),
                    "channels": src.count,
                    "height": src.height,
                    "width": src.width,
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs),
                    "bounds": list(src.bounds),
                    "transform": list(src.transform)[:6],
                    "raw_min": float(arr_s2.min()),
                    "raw_max": float(arr_s2.max()),
                    "raw_mean": float(arr_s2.mean()),
                    "norm_min": float(norm_s2.min()),
                    "norm_max": float(norm_s2.max()),
                    "cloud_fraction": compute_cloud_fraction(mask),
                }
                print(f"Sample Sentinel-2 Optical Metadata:")
                print(f"  Filename:   {sample_s2_meta['file']}")
                print(f"  Dimensions: {sample_s2_meta['channels']} multispectral bands x {sample_s2_meta['height']} x {sample_s2_meta['width']}")
                print(f"  Raw Dtype:  {sample_s2_meta['dtype']} (L1C/L2A DN)")
                print(f"  CRS:        {sample_s2_meta['crs']}")
                print(f"  Raw Range:  [{sample_s2_meta['raw_min']}, {sample_s2_meta['raw_max']}] DN (scale: 10,000)")
                print(f"  Normalized: [{sample_s2_meta['norm_min']:.4f}, {sample_s2_meta['norm_max']:.4f}] reflectance")
                print(f"  Cloud Frac: {sample_s2_meta['cloud_fraction'] * 100:.1f}%")
        except Exception as e:
            corrupted_files.append((sample_s2, str(e)))

    # Spatial alignment check
    print(f"--------------------------------------------------------------------------------")
    if s1_files and s2_files:
        report_align = check_geospatial_alignment(sample_s2_meta, sample_s1_meta)
        alignment_report = {
            "is_aligned": report_align.is_aligned,
            "crs_match": report_align.crs_match,
            "transform_match": report_align.transform_match,
            "shape_match": report_align.shape_match,
            "details": report_align.details,
        }
        print(f"Multimodal Spatial Alignment Verification:")
        print(f"  Shape Match:     {report_align.shape_match}")
        print(f"  CRS Match:       {report_align.crs_match} ({sample_s2_meta.get('crs')} vs {sample_s1_meta.get('crs')})")
        print(f"  Transform Match: {report_align.transform_match}")
        print(f"  Aligned:         {'YES (Valid coregistration)' if report_align.is_aligned else 'NO'}")

    print(f"Corrupted or Unreadable Files: {len(corrupted_files)}")
    print(f"================================================================================")

    full_report = {
        "status": "success",
        "dataset_root": str(root.resolve()),
        "total_tifs": len(all_tifs),
        "s1_count": len(s1_files),
        "s2_count": len(s2_files),
        "rois": sorted(list(rois)),
        "patch_count": len(patches),
        "sample_s1": sample_s1_meta,
        "sample_s2": sample_s2_meta,
        "alignment": alignment_report,
        "corrupted_files": corrupted_files,
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"Inspection report saved to: {out_p.resolve()}")

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SEN12MS-CR-TS satellite dataset.")
    parser.add_argument("--data-dir", default="data/raw/SEN12MSCRTS/extracted", help="Path to extracted dataset")
    parser.add_argument("--output", default="outputs/reports/dataset_inspection.json", help="Path to output JSON")
    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.output)
