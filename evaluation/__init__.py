"""Evaluation package for multimodal satellite image reconstruction."""

from evaluation.baselines import passthrough_baseline, spatial_inpaint_baseline
from evaluation.compare_models import run_benchmark_comparison
from evaluation.evaluate import evaluate_model
from evaluation.metrics import (
    calculate_mae,
    calculate_psnr,
    calculate_rmse,
    calculate_ssim,
    evaluate_sample_metrics,
)

__all__ = [
    "calculate_psnr",
    "calculate_ssim",
    "calculate_mae",
    "calculate_rmse",
    "evaluate_sample_metrics",
    "passthrough_baseline",
    "spatial_inpaint_baseline",
    "evaluate_model",
    "run_benchmark_comparison",
]
