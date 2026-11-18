"""Training pipeline for CNN-only and Optical-only U-Net baselines."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_adapter import SEN12MSCRTSDataset
from evaluation.metrics import calculate_psnr, calculate_ssim
from models.generator import MultimodalUNet
from models.unet_baseline import OpticalOnlyUNet
from training.checkpoint import save_checkpoint
from training.losses import CloudAwareReconstructionLoss
from utils.config import load_yaml_config
from utils.logging_utils import setup_logger
from utils.reproducibility import set_seed


def train_baseline(
    model_type: str = "multimodal_unet",  # "multimodal_unet" or "optical_only_unet"
    config_path: str | Path = "configs/default.yaml",
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    checkpoint_name: str | None = None,
) -> dict:
    config = load_yaml_config(config_path)
    set_seed(config.get("seed", 42))

    device_str = config.hardware.device
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")

    log_name = f"training_{model_type}.log"
    logger = setup_logger(f"train_{model_type}", log_file=f"outputs/reports/{log_name}")
    logger.info(f"Initializing {model_type} baseline training on device: {device}")

    epochs = epochs_override or config.training.epochs
    batch_size = batch_size_override or config.training.batch_size
    optical_channels = config.data.optical_channels
    sar_channels = config.data.sar_channels
    use_mask = config.model.use_mask_input

    # Datasets
    train_dataset = SEN12MSCRTSDataset(
        root_dir=config.data.raw_dir,
        split="train",
        optical_bands=optical_channels,
        sar_bands=sar_channels,
        use_mask=use_mask,
        synthetic_fallback=False,
    )
    val_dataset = SEN12MSCRTSDataset(
        root_dir=config.data.raw_dir,
        split="val",
        optical_bands=optical_channels,
        sar_bands=sar_channels,
        use_mask=use_mask,
        synthetic_fallback=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    # Initialize Architecture
    if model_type == "optical_only_unet":
        model = OpticalOnlyUNet(
            optical_channels=optical_channels,
            use_mask=use_mask,
            base_channels=config.model.base_channels,
        ).to(device)
        ckpt_default = "optical_only_unet.pt"
    else:  # multimodal_unet
        model = MultimodalUNet(
            optical_channels=optical_channels,
            sar_channels=sar_channels,
            use_mask=use_mask,
            base_channels=config.model.base_channels,
        ).to(device)
        ckpt_default = "multimodal_unet.pt"

    checkpoint_filename = checkpoint_name or ckpt_default

    criterion = CloudAwareReconstructionLoss(
        lambda_cloud=config.training.lambda_cloud,
        lambda_clear=config.training.lambda_clear,
        use_mask_weighting=config.training.use_cloud_aware_loss,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr_g,
        betas=(0.9, 0.999),
        weight_decay=config.training.weight_decay,
    )

    history = {
        "epoch": [],
        "loss_train": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_psnr = -1.0
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0

        for batch in train_loader:
            optical = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            mask = batch["cloud_mask"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            if model_type == "optical_only_unet":
                recon = model(optical, mask)
            else:
                recon = model(optical, sar, mask)

            loss, _ = criterion(recon, target, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        mean_loss = train_loss / max(batches, 1)

        # Validation
        val_psnr = 0.0
        val_ssim = 0.0
        val_count = 0

        if epoch % config.training.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for v_batch in val_loader:
                    v_opt = v_batch["optical"].to(device)
                    v_sar = v_batch["sar"].to(device)
                    v_mask = v_batch["cloud_mask"].to(device)
                    v_target = v_batch["target"].to(device)

                    if model_type == "optical_only_unet":
                        v_recon = model(v_opt, v_mask)
                    else:
                        v_recon = model(v_opt, v_sar, v_mask)

                    v_recon_np = np.moveaxis(v_recon.cpu().numpy(), 1, -1)
                    v_target_np = np.moveaxis(v_target.cpu().numpy(), 1, -1)

                    for b in range(v_recon_np.shape[0]):
                        p = calculate_psnr(v_recon_np[b], v_target_np[b])
                        s = calculate_ssim(v_recon_np[b], v_target_np[b])
                        if np.isfinite(p):
                            val_psnr += p
                        if np.isfinite(s):
                            val_ssim += s
                        val_count += 1

            val_psnr = val_psnr / max(val_count, 1)
            val_ssim = val_ssim / max(val_count, 1)

            logger.info(
                f"Epoch [{epoch}/{epochs}] - Loss: {mean_loss:.4f} | "
                f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}"
            )

        history["epoch"].append(epoch)
        history["loss_train"].append(mean_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        checkpoint_state = {
            "epoch": epoch,
            "generator": model.state_dict(),
            "model_type": model_type,
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "config": dict(config),
        }
        save_checkpoint(checkpoint_state, checkpoint_dir, checkpoint_filename, is_best=False)

    elapsed = time.time() - start_time
    logger.info(f"Training for {model_type} completed in {elapsed:.1f}s. Best Val PSNR: {best_psnr:.2f} dB.")

    hist_path = Path("outputs/reports") / f"training_history_{model_type}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models for Satellite Cloud Removal.")
    parser.add_argument("--model-type", default="multimodal_unet", choices=["multimodal_unet", "optical_only_unet"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    train_baseline(
        model_type=args.model_type,
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        checkpoint_name=args.checkpoint,
    )
