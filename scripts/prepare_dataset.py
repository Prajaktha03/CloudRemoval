"""Prepare and export reproducible dataset manifests for training, validation, and testing."""

import argparse
import csv
import os
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.dataset_adapter import SEN12MSCRTSDataset


def prepare_manifests(
    data_dir: str | Path = "data/raw/SEN12MSCRTS/extracted",
    output_dir: str | Path = "data/manifests",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    summary = {}
    
    print(f"================================================================================")
    print(f"Generating Reproducible Multimodal Dataset Manifests")
    print(f"Source Directory: {Path(data_dir).resolve()}")
    print(f"Destination:      {out_path.resolve()}")
    print(f"================================================================================")
    
    for split in splits:
        ds = SEN12MSCRTSDataset(root_dir=data_dir, split=split, synthetic_fallback=False)
        csv_file = out_path / f"manifest_{split}.csv"
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "roi", "patch_id", "cloudy_s2_path", "cloudy_date",
                "sar_s1_path", "sar_date", "clear_s2_path", "clear_date", "cloud_fraction"
            ])
            for rec in ds.records:
                writer.writerow([
                    rec.roi, rec.patch_id, rec.cloudy_s2_path, rec.cloudy_date,
                    rec.sar_s1_path, rec.sar_date, rec.clear_s2_path, rec.clear_date, rec.cloud_fraction
                ])
                
        summary[split] = len(ds.records)
        print(f"  Split '{split}': {len(ds.records)} sample pairs written to {csv_file.name}")
        
    print(f"--------------------------------------------------------------------------------")
    print(f"Total Multimodal Pairs: {sum(summary.values())}")
    print(f"Train / Val / Test Geographic Partition Complete.")
    print(f"================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset manifests.")
    parser.add_argument("--data-dir", default="data/raw/SEN12MSCRTS/extracted", help="Path to extracted dataset")
    parser.add_argument("--output-dir", default="data/manifests", help="Output directory for manifests")
    args = parser.parse_args()
    prepare_manifests(args.data_dir, args.output_dir)
