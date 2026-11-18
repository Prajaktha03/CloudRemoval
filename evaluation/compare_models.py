"""Benchmark comparison across all 5 models and baselines on identical held-out test data."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_adapter import SEN12MSCRTSDataset
from evaluation.baselines import passthrough_baseline, spatial_inpaint_baseline
from evaluation.metrics import evaluate_sample_metrics
from models.generator import MultimodalUNet
from models.unet_baseline import OpticalOnlyUNet
from training.checkpoint import load_checkpoint
from training.losses import composite_cloud_free


def run_benchmark_comparison(
    checkpoints: dict[str, str],
    data_dir: str | Path = "data/raw/SEN12MSCRTS/extracted",
    split: str = "test",
    output_dir: str | Path = "outputs/reports",
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load held-out test dataset
    dataset = SEN12MSCRTSDataset(
        root_dir=data_dir,
        split=split,
        optical_bands=13,
        sar_bands=2,
        use_mask=True,
        synthetic_fallback=True,
    )
    total_samples = len(dataset)
    if max_eval_samples and max_eval_samples < total_samples:
        total_samples = max_eval_samples

    print(f"================================================================================")
    print(f"Scientific Benchmark Comparison on Held-Out Test Set ({split})")
    print(f"Evaluating {total_samples} samples across 5 models/baselines...")
    print(f"================================================================================")

    # Load PyTorch models if checkpoints exist
    loaded_models: dict[str, Any] = {}
    for name, ckpt_file in checkpoints.items():
        p = Path(ckpt_file)
        if p.is_file():
            try:
                ckpt = load_checkpoint(p, device=device)
                cfg = ckpt.get("config", {})
                b_chan = cfg.get("model", {}).get("base_channels", 32)
                if "optical_only" in name:
                    m = OpticalOnlyUNet(optical_channels=13, use_mask=True, base_channels=b_chan).to(device)
                else:
                    m = MultimodalUNet(optical_channels=13, sar_channels=2, use_mask=True, base_channels=b_chan).to(device)
                st = ckpt.get("generator", ckpt.get("model_state_dict", ckpt))
                m.load_state_dict(st)
                m.eval()
                loaded_models[name] = m
                print(f"Loaded model [{name}] from {p.name}")
            except Exception as e:
                print(f"Could not load [{name}] from {p}: {e}")
        else:
            print(f"Checkpoint not found for [{name}]: {ckpt_file} (skipping neural evaluation)")

    # Define model evaluation suite
    model_keys = [
        "A_Passthrough_Control",
        "A_Spatial_Inpainting",
        "B_Optical_Only_UNet",
        "C_Multimodal_UNet_CNN",
        "D_Multimodal_cGAN",
        "E_Multimodal_cGAN_Composite",
    ]

    results: dict[str, dict[str, list[float]]] = {
        k: {"psnr": [], "ssim": [], "mae": [], "rmse": [], "cloud_psnr": [], "clear_psnr": []}
        for k in model_keys
    }

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_eval_samples and idx >= max_eval_samples:
                break

            opt_t = batch["optical"].to(device)
            sar_t = batch["sar"].to(device)
            mask_t = batch["cloud_mask"].to(device)
            target_t = batch["target"].to(device)

            opt_np = np.moveaxis(opt_t[0].cpu().numpy(), 0, -1)
            target_np = np.moveaxis(target_t[0].cpu().numpy(), 0, -1)
            mask_np = np.moveaxis(mask_t[0].cpu().numpy(), 0, -1)

            # 1. Passthrough control
            pred_pass = passthrough_baseline(opt_np)
            m_pass = evaluate_sample_metrics(pred_pass, target_np, mask_np)
            _record_metrics(results["A_Passthrough_Control"], m_pass)

            # 2. Spatial inpainting
            pred_inpaint = spatial_inpaint_baseline(opt_np, mask_np)
            m_inpaint = evaluate_sample_metrics(pred_inpaint, target_np, mask_np)
            _record_metrics(results["A_Spatial_Inpainting"], m_inpaint)

            # 3. Optical-Only UNet
            if "optical_only" in loaded_models:
                m_opt = loaded_models["optical_only"]
                pred_opt_t = m_opt(opt_t, mask_t)
                pred_opt_np = np.moveaxis(pred_opt_t[0].cpu().numpy(), 0, -1)
                m_opt_res = evaluate_sample_metrics(pred_opt_np, target_np, mask_np)
                _record_metrics(results["B_Optical_Only_UNet"], m_opt_res)

            # 4. Multimodal UNet (CNN-only)
            if "multimodal_unet" in loaded_models:
                m_multi = loaded_models["multimodal_unet"]
                pred_multi_t = m_multi(opt_t, sar_t, mask_t)
                pred_multi_np = np.moveaxis(pred_multi_t[0].cpu().numpy(), 0, -1)
                m_multi_res = evaluate_sample_metrics(pred_multi_np, target_np, mask_np)
                _record_metrics(results["C_Multimodal_UNet_CNN"], m_multi_res)

            # 5. Multimodal cGAN
            if "multimodal_cgan" in loaded_models:
                m_gan = loaded_models["multimodal_cgan"]
                pred_gan_t = m_gan(opt_t, sar_t, mask_t)
                pred_gan_np = np.moveaxis(pred_gan_t[0].cpu().numpy(), 0, -1)
                m_gan_res = evaluate_sample_metrics(pred_gan_np, target_np, mask_np)
                _record_metrics(results["D_Multimodal_cGAN"], m_gan_res)

                # 6. Composite cGAN
                pred_comp_t = composite_cloud_free(pred_gan_t, opt_t, mask_t)
                pred_comp_np = np.moveaxis(pred_comp_t[0].cpu().numpy(), 0, -1)
                m_comp_res = evaluate_sample_metrics(pred_comp_np, target_np, mask_np)
                _record_metrics(results["E_Multimodal_cGAN_Composite"], m_comp_res)

    # Compute summary statistics
    summary_table = []
    headers = [
        "Model / Method",
        "PSNR Full (dB)",
        "SSIM Full",
        "MAE Full",
        "RMSE Full",
        "PSNR Cloud (dB)",
        "PSNR Clear (dB)",
        "Sample Count",
    ]

    csv_rows = []
    for k in model_keys:
        d = results[k]
        cnt = len(d["psnr"])
        if cnt == 0:
            continue
        psnr_m, psnr_s = np.mean(d["psnr"]), np.std(d["psnr"])
        ssim_m, ssim_s = np.mean(d["ssim"]), np.std(d["ssim"])
        mae_m = np.mean(d["mae"])
        rmse_m = np.mean(d["rmse"])
        c_psnr_m = np.mean([v for v in d["cloud_psnr"] if np.isfinite(v)]) if d["cloud_psnr"] else float("nan")
        clr_psnr_m = np.mean([v for v in d["clear_psnr"] if np.isfinite(v)]) if d["clear_psnr"] else float("nan")

        row = [
            k.replace("_", " "),
            f"{psnr_m:.2f} ± {psnr_s:.2f}",
            f"{ssim_m:.4f} ± {ssim_s:.4f}",
            f"{mae_m:.4f}",
            f"{rmse_m:.4f}",
            f"{c_psnr_m:.2f}" if np.isfinite(c_psnr_m) else "N/A",
            f"{clr_psnr_m:.2f}" if np.isfinite(clr_psnr_m) else "N/A",
            cnt,
        ]
        summary_table.append(row)
        csv_rows.append({
            "model": k,
            "psnr_mean": psnr_m,
            "psnr_std": psnr_s,
            "ssim_mean": ssim_m,
            "ssim_std": ssim_s,
            "mae_mean": mae_m,
            "rmse_mean": rmse_m,
            "cloud_psnr_mean": c_psnr_m,
            "clear_psnr_mean": clr_psnr_m,
            "samples": cnt,
        })

    # Print markdown table
    print(f"\nBenchmark Results Table:")
    print(f"| " + " | ".join(headers) + " |")
    print(f"| " + " | ".join(["---"] * len(headers)) + " |")
    for r in summary_table:
        print(f"| " + " | ".join(str(x) for x in r) + " |")
    print(f"================================================================================")

    # Save to CSV
    csv_file = out_path / "model_comparison.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["model"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Save to JSON
    json_file = out_path / "model_comparison.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"summary": csv_rows, "split": split, "total_samples": total_samples}, f, indent=2)

    print(f"Comparison saved to {csv_file} and {json_file}")
    return {"summary": csv_rows, "table": summary_table}


def _record_metrics(storage: dict[str, list[float]], metrics: dict[str, Any]) -> None:
    if np.isfinite(metrics["psnr_full"]):
        storage["psnr"].append(metrics["psnr_full"])
    if np.isfinite(metrics["ssim_full"]):
        storage["ssim"].append(metrics["ssim_full"])
    storage["mae"].append(metrics["mae_full"])
    storage["rmse"].append(metrics["rmse_full"])
    if metrics.get("psnr_cloud") is not None and np.isfinite(metrics["psnr_cloud"]):
        storage["cloud_psnr"].append(metrics["psnr_cloud"])
    if metrics.get("psnr_clear") is not None and np.isfinite(metrics["psnr_clear"]):
        storage["clear_psnr"].append(metrics["psnr_clear"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark comparison across all models.")
    parser.add_argument("--gan-ckpt", default="outputs/checkpoints/multimodal_cgan.pt")
    parser.add_argument("--unet-ckpt", default="outputs/checkpoints/multimodal_unet.pt")
    parser.add_argument("--opt-ckpt", default="outputs/checkpoints/optical_only_unet.pt")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    checkpoints = {
        "multimodal_cgan": args.gan_ckpt,
        "multimodal_unet": args.unet_ckpt,
        "optical_only": args.opt_ckpt,
    }

    run_benchmark_comparison(checkpoints, max_eval_samples=args.max_samples)
