"""Unit tests for inference and GeoTIFF export pipelines."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from inference.geotiff_io import read_satellite_image, write_satellite_image
from inference.predict import CloudRemovalPredictor
from models.generator import MultimodalUNet
from training.checkpoint import save_checkpoint


def test_geotiff_io_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "test_optical.tif"
        data = np.random.uniform(0.0, 1.0, (64, 64, 3)).astype(np.float32)

        fake_profile = {
            "driver": "GTiff",
            "height": 64,
            "width": 64,
            "count": 3,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": (0.0001, 0, 10.0, 0, -0.0001, 20.0),
        }

        written_path = write_satellite_image(data, test_path, profile=fake_profile)
        read_arr, read_prof = read_satellite_image(written_path, filename=test_path.name)

        assert read_arr.shape == (64, 64, 3)
        assert read_prof is not None
        assert str(read_prof["crs"]) == "EPSG:4326"


def test_predictor_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "temp_generator.pt"
        model = MultimodalUNet(optical_channels=13, sar_channels=2, use_mask=True, base_channels=16)

        save_checkpoint(
            {
                "generator": model.state_dict(),
                "model_type": "multimodal_unet",
                "config": {"data": {"optical_channels": 13, "sar_channels": 2}, "model": {"base_channels": 16, "use_mask_input": True}},
            },
            tmp_dir,
            "temp_generator.pt",
        )

        predictor = CloudRemovalPredictor(ckpt_path)

        dummy_opt = np.random.uniform(0.0, 1.0, (64, 64, 13)).astype(np.float32)
        dummy_sar = np.random.uniform(0.0, 1.0, (64, 64, 2)).astype(np.float32)
        dummy_ref = np.random.uniform(0.0, 1.0, (64, 64, 13)).astype(np.float32)

        out = predictor.predict(dummy_opt, dummy_sar, reference=dummy_ref, use_compositing=True)

        assert out.reconstruction.shape == (64, 64, 13)
        assert out.cloud_mask.shape == (64, 64, 1)
        assert out.metrics is not None
        assert "psnr_full" in out.metrics
