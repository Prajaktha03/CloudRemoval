"""Dataset adapter for SEN12MS-CR-TS multimodal satellite observations."""

from collections import defaultdict
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from rasterio.errors import RasterioIOError

from datasets.cloud_masks import compute_cloud_fraction, detect_s2_clouds
from datasets.preprocessing import normalize_optical, normalize_sar, read_geotiff_multimodal


@dataclass
class MultimodalSampleRecord:
    roi: str
    patch_id: str
    cloudy_s2_path: str
    cloudy_date: str
    sar_s1_path: str
    sar_date: str
    clear_s2_path: str
    clear_date: str
    cloud_fraction: float


class SEN12MSCRTSDataset(Dataset):
    """PyTorch Dataset adapter for SEN12MS-CR-TS multimodal multitemporal satellite data."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        optical_bands: int = 13,
        sar_bands: int = 2,
        use_mask: bool = True,
        min_cloud_fraction: float = 0.05,
        max_clear_cloud_fraction: float = 0.10,
        synthetic_fallback: bool = True,
        synthetic_samples: int = 20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.optical_bands = optical_bands
        self.sar_bands = sar_bands
        self.use_mask = use_mask
        self.min_cloud_fraction = min_cloud_fraction
        self.max_clear_cloud_fraction = max_clear_cloud_fraction
        self.synthetic_fallback = synthetic_fallback
        self.synthetic_samples = synthetic_samples
        self.seed = seed
        self.is_synthetic = False

        self.records: list[MultimodalSampleRecord] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Scan directory and index paired cloudy S2, S1 SAR, and clear reference S2."""
        if not self.root_dir.is_dir():
            if self.synthetic_fallback:
                self._generate_synthetic_manifest()
                return
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root_dir}")

        all_tifs = glob.glob(str(self.root_dir / "**" / "*.tif"), recursive=True)
        if not all_tifs:
            if self.synthetic_fallback:
                self._generate_synthetic_manifest()
                return
            raise ValueError(f"No GeoTIFF (.tif) files found under {self.root_dir}")

        # SEN12MS-CR-TS filename pattern:
        # s1_ROIs2017_22_ImgNo_16_2018-07-18_patch_0.tif
        pattern = re.compile(r"(s1|s2)_ROIs(\d+)_(\d+)_ImgNo_(\d+)_([\d-]+)_patch_(\d+)\.tif")

        # Organize by ROI and patch_id
        grouped: dict[str, dict[str, dict[str, list[tuple[int, str, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for p in all_tifs:
            fname = os.path.basename(p)
            match = pattern.match(fname)
            if match:
                sensor, year, roi_id, img_no, date, patch_id = match.groups()
                roi_key = f"ROI_{year}_{roi_id}"
                grouped[roi_key][patch_id][sensor].append((int(img_no), date, p))

        # Split ROIs geographically:
        # ROI_2017_22 -> train and val
        # ROI_2017_140 -> test
        candidate_rois = list(grouped.keys())

        # Build candidate records
        indexed_records: list[MultimodalSampleRecord] = []

        for roi_key, patches in grouped.items():
            patch_keys = sorted(patches.keys(), key=lambda x: int(x) if x.isdigit() else x)
            # Geographic / Patch partition for train/val/test
            if "140" in roi_key:
                # Test ROI
                if self.split != "test":
                    continue
                assigned_patches = patch_keys
            else:
                # Train/Val ROI (e.g. ROI_2017_22)
                if self.split == "test":
                    continue
                split_idx = int(0.8 * len(patch_keys))
                if self.split == "train":
                    assigned_patches = patch_keys[:split_idx]
                else:  # "val"
                    assigned_patches = patch_keys[split_idx:]

            for patch_id in assigned_patches:
                sensor_dict = patches[patch_id]
                s1_list = sorted(sensor_dict.get("s1", []), key=lambda x: x[0])
                s2_list = sorted(sensor_dict.get("s2", []), key=lambda x: x[0])

                if not s1_list or not s2_list:
                    continue

                if len(s2_list) < 2:
                    continue

                observations: list[tuple[tuple[int, str, str], float, float]] = []
                for candidate in s2_list:
                    try:
                        raw_optical, _ = read_geotiff_multimodal(candidate[2])
                    except RasterioIOError as exc:
                        warnings.warn(
                            f"Skipping unreadable Sentinel-2 acquisition {candidate[2]}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    optical = normalize_optical(raw_optical)
                    if self.optical_bands == 3:
                        optical = optical[..., [3, 2, 1]]
                    elif self.optical_bands < optical.shape[-1]:
                        optical = optical[..., : self.optical_bands]

                    cloud_fraction = compute_cloud_fraction(detect_s2_clouds(optical))
                    visible_bands = [3, 2, 1] if optical.shape[-1] >= 4 else slice(None, 3)
                    visible_brightness = float(np.mean(optical[..., visible_bands]))
                    observations.append((candidate, cloud_fraction, visible_brightness))

                cloudy_candidates = [
                    observation
                    for observation in observations
                    if observation[1] >= self.min_cloud_fraction
                ]
                if not cloudy_candidates:
                    continue
                cloudy_observation = max(
                    cloudy_candidates,
                    key=lambda observation: (observation[1], observation[2]),
                )
                clear_candidates = [
                    observation
                    for observation in observations
                    if observation[0][2] != cloudy_observation[0][2]
                    and observation[1] <= self.max_clear_cloud_fraction
                ]
                if not clear_candidates:
                    continue
                clear_observation = min(
                    clear_candidates,
                    key=lambda observation: (observation[1], observation[2]),
                )
                cloudy_candidate = cloudy_observation[0]
                clear_candidate = clear_observation[0]

                cloudy_img_no = cloudy_candidate[0]
                matched_s1 = None
                for sar_candidate in sorted(s1_list, key=lambda s1: abs(s1[0] - cloudy_img_no)):
                    try:
                        read_geotiff_multimodal(sar_candidate[2])
                    except RasterioIOError as exc:
                        warnings.warn(
                            f"Skipping unreadable Sentinel-1 acquisition {sar_candidate[2]}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    matched_s1 = sar_candidate
                    break
                if matched_s1 is None:
                    continue

                indexed_records.append(
                    MultimodalSampleRecord(
                        roi=roi_key,
                        patch_id=patch_id,
                        cloudy_s2_path=cloudy_candidate[2],
                        cloudy_date=cloudy_candidate[1],
                        sar_s1_path=matched_s1[2],
                        sar_date=matched_s1[1],
                        clear_s2_path=clear_candidate[2],
                        clear_date=clear_candidate[1],
                        cloud_fraction=cloudy_observation[1],
                    )
                )

        if not indexed_records:
            if self.synthetic_fallback:
                self._generate_synthetic_manifest()
                return
            raise ValueError(f"No valid multimodal pairs found for split '{self.split}'.")

        self.records = indexed_records

    def _generate_synthetic_manifest(self) -> None:
        """Create synthetic dataset records for smoke-testing and offline environments."""
        self.is_synthetic = True
        rng = np.random.default_rng(self.seed + (0 if self.split == "train" else 1 if self.split == "val" else 2))
        self.records = []
        for i in range(self.synthetic_samples):
            self.records.append(
                MultimodalSampleRecord(
                    roi=f"SYNTH_ROI_{self.split}",
                    patch_id=str(i),
                    cloudy_s2_path=f"synthetic://cloudy_{i}.tif",
                    cloudy_date="2018-07-15",
                    sar_s1_path=f"synthetic://sar_{i}.tif",
                    sar_date="2018-07-16",
                    clear_s2_path=f"synthetic://clear_{i}.tif",
                    clear_date="2018-01-10",
                    cloud_fraction=float(rng.uniform(0.15, 0.65)),
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def _generate_synthetic_item(self, record: MultimodalSampleRecord, idx: int) -> dict[str, Any]:
        """Generate physically consistent synthetic satellite data tensors."""
        rng = np.random.default_rng(self.seed + idx * 31)
        h, w = 256, 256
        c_opt = self.optical_bands
        c_sar = self.sar_bands

        # Underlying clean ground reflectance [0, 1]
        base_reflectance = rng.uniform(0.05, 0.45, size=(h, w, c_opt)).astype(np.float32)

        # Cloud mask: Perlin-like elliptical cloud clusters
        yy, xx = np.mgrid[:h, :w]
        cy, cx = rng.integers(60, 196, size=2)
        radius = rng.uniform(40, 80)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        cloud_presence = (dist < radius).astype(np.float32)

        # Cloud overlay: bright diffuse scattering
        cloudy_optical = base_reflectance.copy()
        cloudy_optical += cloud_presence[..., np.newaxis] * rng.uniform(0.4, 0.6)
        cloudy_optical = np.clip(cloudy_optical, 0.0, 1.0)

        # SAR radar backscatter: structural edges, completely unaffected by clouds
        sar_backscatter = rng.uniform(0.1, 0.8, size=(h, w, c_sar)).astype(np.float32)

        cloud_mask = cloud_presence[..., np.newaxis]

        # Convert to CHW PyTorch tensors
        return {
            "optical": torch.from_numpy(np.moveaxis(cloudy_optical, -1, 0)),
            "sar": torch.from_numpy(np.moveaxis(sar_backscatter, -1, 0)),
            "cloud_mask": torch.from_numpy(np.moveaxis(cloud_mask, -1, 0)),
            "target": torch.from_numpy(np.moveaxis(base_reflectance, -1, 0)),
            "metadata": {
                "roi": record.roi,
                "patch_id": record.patch_id,
                "cloudy_date": record.cloudy_date,
                "clear_date": record.clear_date,
                "cloud_fraction": float(np.mean(cloud_presence)),
                "is_synthetic": True,
            },
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        if self.is_synthetic:
            return self._generate_synthetic_item(record, idx)

        raw_cloudy, _ = read_geotiff_multimodal(record.cloudy_s2_path)
        raw_sar, _ = read_geotiff_multimodal(record.sar_s1_path)
        raw_clear, _ = read_geotiff_multimodal(record.clear_s2_path)

        # Normalization
        optical = normalize_optical(raw_cloudy)
        sar = normalize_sar(raw_sar)
        clear = normalize_optical(raw_clear)

        # Channel selection if requested
        if self.optical_bands == 3:
            optical = optical[..., [3, 2, 1]]  # RGB: B4, B3, B2
            clear = clear[..., [3, 2, 1]]
        elif self.optical_bands < optical.shape[-1]:
            optical = optical[..., : self.optical_bands]
            clear = clear[..., : self.optical_bands]

        if self.sar_bands < sar.shape[-1]:
            sar = sar[..., : self.sar_bands]

        # Cloud mask detection
        mask = detect_s2_clouds(optical)
        cloud_frac = compute_cloud_fraction(mask)

        return {
            "optical": torch.from_numpy(np.moveaxis(optical, -1, 0)),
            "sar": torch.from_numpy(np.moveaxis(sar, -1, 0)),
            "cloud_mask": torch.from_numpy(np.moveaxis(mask, -1, 0)),
            "target": torch.from_numpy(np.moveaxis(clear, -1, 0)),
            "metadata": {
                "roi": record.roi,
                "patch_id": record.patch_id,
                "cloudy_date": record.cloudy_date,
                "clear_date": record.clear_date,
                "cloud_fraction": cloud_frac,
                "is_synthetic": False,
            },
        }
