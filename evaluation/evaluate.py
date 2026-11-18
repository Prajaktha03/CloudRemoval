"""Evaluation pipeline for trained models on held-out test data."""

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
from evaluation.metrics import evaluate_sample_metrics
from models.generator import MultimodalUNet
from models.unet_baseline import OpticalOnlyUNet
from training.checkpoint import load_checkpoint
from training.losses import composite_cloud_free
from utils.config import load_yaml_config


def evaluate_model(
    checkpoint_path: str | Path,
    model_type: str = "multimodal_cgan",
    split: str = "test",
    data_dir: str | Path = "data/raw/SEN12MSCRTS/extracted",
    output_dir: str | Path = "outputs/reports",
    device_str: str = "auto",
    use_compositing: bool = False,
) -> dict[str, Any]:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    checkpoint = load_checkpoint(ckpt_p, device=device)
    config_dict = checkpoint.get("config", {})

    optical_channels = config_dict.get("data", {}).get("optical_channels", 13)
    sar_channels = config_dict.get("data", {}).get("sar_channels", 2)
    use_mask = config_dict.get("model", {}).get("use_mask_input", True)
    base_channels = config_dict.get("model", {}).get("base_channels", 32)

    # Initialize model
    if model_type == "optical_only_unet":
        model = OpticalOnlyUNet(
            optical_channels=optical_channels,
            use_mask=use_mask,
            base_channels=base_channels,
        ).to(device)
    else:
        model = MultimodalUNet(
            optical_channels=optical_channels,
            sar_channels=sar_channels,
            use_mask=use_mask,
            base_channels=base_channels,
        ).to(device)

    state = checkpoint.get("generator", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()

    # Load test dataset
    dataset = SEN12MSCRTSDataset(
        root_dir=data_dir,
        split=split,
        optical_bands=optical_channels,
        sar_bands=sar_channels,
        use_mask=use_mask,
        synthetic_fallback=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    records = []
    full_psnr, full_ssim, full_mae, full_rmse = [], [], [], []
    cloud_psnr, cloud_mae, cloud_rmse = [], [], []
    clear_psnr, clear_mae, clear_rmse = []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / f"eval_details_{ckpt_p.stem}.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "roi", "patch_id", "psnr_full", "ssim_full", "mae_full", "rmse_full",
            "psnr_cloud", "mae_cloud", "rmse_cloud", "psnr_clear", "mae_clear", "rmse_clear",
            "cloud_fraction"
        ])

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                opt = batch["optical"].to(device)
                sar = batch["sar"].to(device)
                mask = batch["cloud_mask"].to(device)
                target = batch["target"].to(device)
                meta = batch["metadata"]

                if model_type == "optical_only_unet":
                    pred = model(opt, mask)
                else:
                    pred = model(opt, sar, mask)

                if use_compositing:
                    pred = composite_cloud_free(pred, opt, mask)

                pred_np = np.moveaxis(pred[0].cpu().numpy(), 0, -1)
                target_np = np.moveaxis(target[0].cpu().numpy(), 0, -1)
                mask_np = np.moveaxis(mask[0].cpu().numpy(), 0, -1)

                m = evaluate_sample_metrics(pred_np, target_np, cloud_mask=mask_np)

                if np.isfinite(m["psnr_full"]):
                    full_psnr.append(m["psnr_full"])
                if np.isfinite(m["ssim_full"]):
                    full_ssim.append(m["ssim_full"])
                full_mae.append(m["mae_full"])
                full_rmse.append(m["rmse_full"])

                if m.get("psnr_cloud") is not None and np.isfinite(m["psnr_cloud"]):
                    cloud_psnr.append(m["psnr_cloud"])
                    cloud_mae.append(m["mae_cloud"])
                    cloud_rmse.append(m["rmse_cloud"])

                if m.get("psnr_clear") is not None and np.isfinite(m["psnr_clear"]):
                    clear_psnr.append(m["psnr_clear"])
                    clear_mae.append(m["mae_clear"])
                    clear_rmse.append(m["rmse_clear"])

                writer.writerow([
                    idx,
                    meta["roi"][0],
                    meta["patch_id"][0],
                    m["psnr_full"],
                    m["ssim_full"],
                    m["mae_full"],
                    m["rmse_full"],
                    m.get("psnr_cloud"),
                    m.get("mae_cloud"),
                    m.get("rmse_cloud"),
                    m.get("psnr_clear"),
                    m.get("mae_clear"),
                    m.get("rmse_clear"),
                    meta["cloud_fraction"][0].item() if torch.is_tensor(meta["cloud_fraction"][0]) else meta["cloud_fraction"][0],
                ])

    summary = {
        "model_checkpoint": str(ckpt_p.resolve()),
        "model_type": model_type,
        "split": split,
        "sample_count": len(dataset),
        "compositing": use_compositing,
        "metrics": {
            "psnr_mean": float(np.mean(full_psnr)) if full_psnr else float("nan"),
            "psnr_std": float(np.std(full_psnr)) if full_psnr else float("nan"),
            "ssim_mean": float(np.mean(full_ssim)) if full_ssim else float("nan"),
            "ssim_std": float(np.std(full_ssim)) if full_ssim else float("nan"),
            "mae_mean": float(np.mean(full_mae)) if full_mae else float("nan"),
            "mae_std": float(np.std(full_mae)) if full_mae else float("nan"),
            "rmse_mean": float(np.mean(full_rmse)) if full_rmse else float("nan"),
            "rmse_std": float(np.std(full_rmse)) if full_rmse else float("nan"),
            "cloud_psnr_mean": float(np.mean(cloud_psnr)) if cloud_psnr else float("nan"),
            "cloud_ssim_mean": float("nan"),
            "cloud_mae_mean": float(np.mean(cloud_mae)) if cloud_mae else float("nan"),
            "clear_psnr_mean": float(np.mean(clear_psnr)) if clear_psnr else float("nan"),
            "clear_mae_mean": float(np.mean(clear_mae)) if clear_mae else float("nan"),
        },
        "details_csv": str(csv_file.resolve()),
    }

    json_file = out_path / f"eval_summary_{ckpt_p.stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained satellite cloud removal model.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--model-type", default="multimodal_cgan")
    parser.add_argument("--split", default="test")
    parser.add_argument("--composite", action="store_true")
    args = parser.parse_args()

    res = evaluate_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        split=args.split,
        use_compositing=args.composite,
    )
    print("Evaluation summary:")
    print(json.dumps(res, indent=2))
