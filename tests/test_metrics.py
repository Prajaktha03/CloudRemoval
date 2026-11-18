"""Unit tests for scientific evaluation metrics."""

import math
import numpy as np
import pytest

from evaluation.metrics import (
    calculate_mae,
    calculate_psnr,
    calculate_rmse,
    calculate_ssim,
    evaluate_sample_metrics,
)


def test_identical_images_psnr_and_ssim():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    assert calculate_psnr(img, img) == float("inf")
    assert abs(calculate_ssim(img, img) - 1.0) < 1e-6
    assert calculate_mae(img, img) == 0.0
    assert calculate_rmse(img, img) == 0.0


def test_known_noise_psnr():
    ref = np.zeros((64, 64, 1), dtype=np.float32)
    # Add constant error of 0.1 -> MSE = 0.01 -> PSNR = 10 * log10(1 / 0.01) = 20 dB
    noisy = ref + 0.1
    psnr = calculate_psnr(noisy, ref, data_range=1.0)
    assert abs(psnr - 20.0) < 1e-4
    assert abs(calculate_mae(noisy, ref) - 0.1) < 1e-6
    assert abs(calculate_rmse(noisy, ref) - 0.1) < 1e-6


def test_cloud_region_evaluation():
    ref = np.zeros((32, 32, 3), dtype=np.float32)
    rec = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.zeros((32, 32, 1), dtype=np.float32)

    # Cloud in upper-left 16x16 with error 0.2
    mask[:16, :16, 0] = 1.0
    rec[:16, :16] = 0.2

    metrics = evaluate_sample_metrics(rec, ref, cloud_mask=mask)
    assert metrics["psnr_cloud"] is not None
    # Cloud region MSE = 0.04 -> PSNR = 10 * log10(1 / 0.04) = 13.9794 dB
    assert abs(metrics["psnr_cloud"] - 13.9794) < 1e-3
    # Clear region error = 0 -> PSNR = inf
    assert metrics["psnr_clear"] == float("inf")
