"""Scientific visualization utilities for remote sensing and multimodal cloud removal."""

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from datasets.preprocessing import make_rgb_preview, make_sar_preview


def plot_reconstruction_results(
    cloudy_optical: np.ndarray,
    sar: np.ndarray,
    cloud_mask: np.ndarray,
    reconstruction: np.ndarray,
    reference: np.ndarray | None = None,
    save_path: str | Path | None = None,
    title: str = "Multimodal Deep Learning Satellite Cloud Removal",
) -> plt.Figure:
    """Generate a publication-grade multi-panel figure comparing inputs, mask, reconstruction, and error."""
    n_cols = 6 if reference is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.5), dpi=150)
    if n_cols == 1:
        axes = [axes]

    # 1. Cloudy Optical
    rgb_cloudy = make_rgb_preview(cloudy_optical)
    axes[0].imshow(rgb_cloudy)
    axes[0].set_title("(a) Cloudy Sentinel-2 (RGB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Sentinel-1 SAR
    sar_preview = make_sar_preview(sar)
    axes[1].imshow(sar_preview)
    axes[1].set_title("(b) Sentinel-1 SAR (Dual-Pol)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Cloud Mask
    mask_2d = cloud_mask[..., 0] if cloud_mask.ndim == 3 else cloud_mask
    axes[2].imshow(mask_2d, cmap="Blues", vmin=0, vmax=1)
    cloud_frac = float(np.mean(mask_2d > 0.5)) * 100.0
    axes[2].set_title(f"(c) Cloud Mask ({cloud_frac:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Reconstructed Optical
    rgb_recon = make_rgb_preview(reconstruction)
    axes[3].imshow(rgb_recon)
    axes[3].set_title("(d) Reconstructed Optical", fontsize=11, fontweight="bold", color="darkgreen")
    axes[3].axis("off")

    if reference is not None:
        # 5. Clear Ground Truth Reference
        rgb_ref = make_rgb_preview(reference)
        axes[4].imshow(rgb_ref)
        axes[4].set_title("(e) Clear Reference (Target)", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        # 6. Absolute Error / Difference Heatmap
        diff = np.mean(np.abs(reconstruction - reference), axis=-1)
        im_err = axes[5].imshow(diff, cmap="magma", vmin=0.0, vmax=0.25)
        axes[5].set_title("(f) Reconstruction MAE Heatmap", fontsize=11, fontweight="bold")
        axes[5].axis("off")
        cbar = fig.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04)
        cbar.set_label("Absolute Error", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Generator loss, Discriminator loss, L1 loss, and Validation PSNR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    epochs = history.get("epoch", list(range(1, len(history.get("loss_g", [])) + 1)))

    if "loss_g" in history and "loss_d" in history:
        ax1.plot(epochs, history["loss_g"], label="Generator Loss ($L_G$)", color="#1f77b4", lw=2)
        ax1.plot(epochs, history["loss_d"], label="Discriminator Loss ($L_D$)", color="#ff7f0e", lw=2)
        if "loss_l1" in history:
            ax1.plot(epochs, history["loss_l1"], label="Reconstruction ($L_1$)", color="#2ca02c", linestyle="--")
        ax1.set_title("Training Loss Convergence", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)
    elif "loss_train" in history:
        ax1.plot(epochs, history["loss_train"], label="Train L1 Loss", color="#1f77b4", lw=2)
        ax1.set_title("Training Loss Convergence", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

    if "val_psnr" in history and any(np.isfinite(history["val_psnr"])):
        val_psnr = history["val_psnr"]
        ax2.plot(epochs, val_psnr, label="Validation PSNR (dB)", color="#d62728", lw=2, marker="o")
        ax2.set_title("Validation Quality (PSNR)", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("PSNR (dB)")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_benchmark_bars(
    summary_data: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate side-by-side bar chart of PSNR and SSIM across model variants."""
    models = [d["model"].replace("_", " ") for d in summary_data]
    psnr_vals = [d["psnr_mean"] for d in summary_data]
    psnr_errs = [d.get("psnr_std", 0.0) for d in summary_data]
    ssim_vals = [d["ssim_mean"] for d in summary_data]
    ssim_errs = [d.get("ssim_std", 0.0) for d in summary_data]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # PSNR Bar Chart
    ax1.bar(x, psnr_vals, width, yerr=psnr_errs, capsize=4, color="#3470a3", edgecolor="black")
    ax1.set_ylabel("PSNR (dB) [Higher is better]", fontweight="bold")
    ax1.set_title("Peak Signal-to-Noise Ratio (PSNR)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.7)

    # SSIM Bar Chart
    ax2.bar(x, ssim_vals, width, yerr=ssim_errs, capsize=4, color="#388e3c", edgecolor="black")
    ax2.set_ylabel("SSIM [0 to 1, Higher is better]", fontweight="bold")
    ax2.set_title("Structural Similarity Index (SSIM)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)

    plt.tight_layout()
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig
