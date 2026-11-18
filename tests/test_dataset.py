"""Unit tests for dataset loading, normalization, and cloud masks."""

import numpy as np
import pytest
import torch

from datasets.cloud_masks import compute_cloud_fraction, detect_s2_clouds
from datasets.dataset_adapter import SEN12MSCRTSDataset
from datasets.preprocessing import normalize_optical, normalize_sar


def test_normalization_ranges():
    # Optical uint16 DN to [0, 1]
    raw_opt = np.array([[[1000, 5000, 10000, 15000]]], dtype=np.uint16)
    norm_opt = normalize_optical(raw_opt)
    assert norm_opt.min() == 0.1
    assert norm_opt.max() == 1.0  # clipped to 1.0

    # SAR dB to [0, 1]
    raw_sar = np.array([[[-35.0, -17.5, 0.0, 5.0]]], dtype=np.float32)
    norm_sar = normalize_sar(raw_sar)
    assert abs(norm_sar[0, 0, 0] - 0.0) < 1e-4
    assert abs(norm_sar[0, 0, 1] - 0.5) < 1e-4
    assert abs(norm_sar[0, 0, 2] - 1.0) < 1e-4
    assert abs(norm_sar[0, 0, 3] - 1.0) < 1e-4


def test_cloud_detection_and_fraction():
    # Clear image
    clear_img = np.full((32, 32, 13), 0.10, dtype=np.float32)
    mask_clear = detect_s2_clouds(clear_img)
    assert compute_cloud_fraction(mask_clear) == 0.0

    # Cloudy image with bright blue and cirrus
    cloudy_img = clear_img.copy()
    cloudy_img[8:24, 8:24, 1] = 0.45  # Blue band
    cloudy_img[8:24, 8:24, 2] = 0.45  # Green
    cloudy_img[8:24, 8:24, 3] = 0.45  # Red
    cloudy_img[8:24, 8:24, 10] = 0.05 # Cirrus band

    mask_cloud = detect_s2_clouds(cloudy_img)
    frac = compute_cloud_fraction(mask_cloud)
    assert frac > 0.20


def test_synthetic_dataset_sample():
    ds = SEN12MSCRTSDataset("dummy_path", split="train", synthetic_fallback=True, synthetic_samples=3)
    assert len(ds) == 3
    sample = ds[0]
    assert sample["optical"].shape == (13, 256, 256)
    assert sample["sar"].shape == (2, 256, 256)
    assert sample["cloud_mask"].shape == (1, 256, 256)
    assert sample["target"].shape == (13, 256, 256)
    assert sample["metadata"]["is_synthetic"] is True


def test_real_dataset_uses_cloudiest_and_clearest_acquisitions(tmp_path, monkeypatch):
    import datasets.dataset_adapter as dataset_adapter
    from rasterio.errors import RasterioIOError

    s2_dir = tmp_path / "ROIs2017" / "140" / "S2"
    s1_dir = tmp_path / "ROIs2017" / "140" / "S1"
    s2_dir.mkdir(parents=True)
    s1_dir.mkdir(parents=True)

    paths = {}
    for image_number in (0, 1, 2):
        path = s2_dir / f"s2_ROIs2017_140_ImgNo_{image_number}_2018-01-0{image_number + 1}_patch_0.tif"
        path.touch()
        paths[str(path)] = (0.1, 0.2, 0.3)[image_number]
    for image_number in (1, 2):
        (s1_dir / f"s1_ROIs2017_140_ImgNo_{image_number}_2018-01-0{image_number + 1}_patch_0.tif").touch()

    def fake_read(path):
        if str(path).lower().find("\\s1\\") >= 0:
            if "ImgNo_2_" in str(path):
                raise RasterioIOError("corrupt SAR acquisition")
            return np.zeros((8, 8, 2), dtype=np.float32), {}
        if "ImgNo_1_" in str(path):
            raise RasterioIOError("corrupt optical acquisition")
        return np.full((8, 8, 13), paths[str(path)], dtype=np.float32), {}

    monkeypatch.setattr(dataset_adapter, "read_geotiff_multimodal", fake_read)
    def fake_cloud_mask(optical):
        fraction = {1: 0.0, 2: 0.8, 3: 0.8}[round(float(optical[0, 0, 0]) * 10)]
        mask = np.zeros((optical.shape[0], optical.shape[1], 1), dtype=np.float32)
        mask.reshape(-1)[: round(mask.size * fraction)] = 1.0
        return mask

    monkeypatch.setattr(dataset_adapter, "detect_s2_clouds", fake_cloud_mask)

    with pytest.warns(RuntimeWarning) as warnings_record:
        dataset = SEN12MSCRTSDataset(tmp_path, split="test", synthetic_fallback=False)

    assert any("Skipping unreadable Sentinel-2 acquisition" in str(w.message) for w in warnings_record)
    assert any("Skipping unreadable Sentinel-1 acquisition" in str(w.message) for w in warnings_record)
    assert len(dataset) == 1
    record = dataset.records[0]
    assert "ImgNo_2_" in record.cloudy_s2_path
    assert "ImgNo_0_" in record.clear_s2_path
    assert "ImgNo_2_" not in record.sar_s1_path
    assert record.cloud_fraction == pytest.approx(0.8, abs=1 / 64)
