"""Training pipeline for Multimodal cGAN Satellite Cloud Removal."""

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
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_adapter import SEN12MSCRTSDataset
from evaluation.metrics import calculate_psnr, calculate_ssim
from models.discriminator import PatchGANDiscriminator
from models.generator import MultimodalUNet
from training.checkpoint import save_checkpoint
from training.losses import CloudAwareReconstructionLoss, GANLoss
from utils.config import load_yaml_config
from utils.logging_utils import setup_logger
from utils.reproducibility import set_seed


def train_gan(
    config_path: str | Path = "configs/default.yaml",
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    checkpoint_name: str = "multimodal_cgan.pt",
) -> dict:
    config = load_yaml_config(config_path)
    set_seed(config.get("seed", 42))

    device_str = config.hardware.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger = setup_logger("train_gan", log_file="outputs/reports/training_gan.log")
    logger.info(f"Initializing Multimodal cGAN Training on device: {device}")

    epochs = epochs_override or config.training.epochs
    batch_size = batch_size_override or config.training.batch_size
    optical_channels = config.data.optical_channels
    sar_channels = config.data.sar_channels
    use_mask = config.model.use_mask_input

    # Datasets and Loaders
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=False,
    )

    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Initialize Networks
    net_g = MultimodalUNet(
        optical_channels=optical_channels,
        sar_channels=sar_channels,
        use_mask=use_mask,
        base_channels=config.model.base_channels,
    ).to(device)

    net_d = PatchGANDiscriminator(
        optical_channels=optical_channels,
        sar_channels=sar_channels,
        use_mask=use_mask,
        base_channels=config.model.base_channels,
    ).to(device)

    # Losses & Optimizers
    criterion_gan = GANLoss()
    criterion_recon = CloudAwareReconstructionLoss(
        lambda_cloud=config.training.lambda_cloud,
        lambda_clear=config.training.lambda_clear,
        use_mask_weighting=config.training.use_cloud_aware_loss,
    )

    opt_g = torch.optim.Adam(
        net_g.parameters(),
        lr=config.training.lr_g,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )
    opt_d = torch.optim.Adam(
        net_d.parameters(),
        lr=config.training.lr_d,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )

    lambda_l1 = config.training.lambda_l1
    lambda_adv = config.training.lambda_adv

    history = {
        "epoch": [],
        "loss_d": [],
        "loss_g": [],
        "loss_l1": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_psnr = -1.0
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        net_g.train()
        net_d.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_l1_loss = 0.0
        batches = 0

        for batch in train_loader:
            optical = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            mask = batch["cloud_mask"].to(device)
            target = batch["target"].to(device)

            # ---------------------
            # 1. Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            # Forward generator
            fake_optical = net_g(optical, sar, mask)

            # Real predictions
            pred_real = net_d(optical, sar, target, mask)
            loss_d_real = criterion_gan(pred_real, True)

            # Fake predictions
            pred_fake = net_d(optical, sar, fake_optical.detach(), mask)
            loss_d_fake = criterion_gan(pred_fake, False)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # -----------------
            # 2. Train Generator
            # -----------------
            opt_g.zero_grad()
            # Generator adversarial loss
            pred_fake_g = net_d(optical, sar, fake_optical, mask)
            loss_g_adv = criterion_gan(pred_fake_g, True)

            # Reconstruction loss (cloud-aware L1)
            loss_recon, loss_dict = criterion_recon(fake_optical, target, mask)

            loss_g = lambda_adv * loss_g_adv + lambda_l1 * loss_recon
            loss_g.backward()
            opt_g.step()

            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            epoch_l1_loss += loss_recon.item()
            batches += 1

        mean_d = epoch_d_loss / max(batches, 1)
        mean_g = epoch_g_loss / max(batches, 1)
        mean_l1 = epoch_l1_loss / max(batches, 1)

        # -----------------
        # 3. Validation Loop
        # -----------------
        val_psnr = 0.0
        val_ssim = 0.0
        val_count = 0

        if epoch % config.training.val_interval == 0:
            net_g.eval()
            with torch.no_grad():
                for v_batch in val_loader:
                    v_opt = v_batch["optical"].to(device)
                    v_sar = v_batch["sar"].to(device)
                    v_mask = v_batch["cloud_mask"].to(device)
                    v_target = v_batch["target"].to(device)

                    v_fake = net_g(v_opt, v_sar, v_mask)

                    v_fake_np = np.moveaxis(v_fake.cpu().numpy(), 1, -1)
                    v_target_np = np.moveaxis(v_target.cpu().numpy(), 1, -1)

                    for b in range(v_fake_np.shape[0]):
                        p = calculate_psnr(v_fake_np[b], v_target_np[b])
                        s = calculate_ssim(v_fake_np[b], v_target_np[b])
                        if np.isfinite(p):
                            val_psnr += p
                        if np.isfinite(s):
                            val_ssim += s
                        val_count += 1

            val_psnr = val_psnr / max(val_count, 1)
            val_ssim = val_ssim / max(val_count, 1)

            logger.info(
                f"Epoch [{epoch}/{epochs}] - Loss D: {mean_d:.4f} | Loss G: {mean_g:.4f} | "
                f"L1: {mean_l1:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}"
            )

        history["epoch"].append(epoch)
        history["loss_d"].append(mean_d)
        history["loss_g"].append(mean_g)
        history["loss_l1"].append(mean_l1)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # Checkpoint saving
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        checkpoint_state = {
            "epoch": epoch,
            "generator": net_g.state_dict(),
            "discriminator": net_d.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "best_psnr": best_psnr,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "config": dict(config),
        }
        save_checkpoint(checkpoint_state, checkpoint_dir, checkpoint_name, is_best=is_best)

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s. Best Validation PSNR: {best_psnr:.2f} dB.")

    # Save history
    hist_path = Path("outputs/reports") / "training_history_gan.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal cGAN for Satellite Cloud Removal.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--checkpoint", default="multimodal_cgan.pt", help="Checkpoint filename")
    args = parser.parse_args()

    train_gan(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        checkpoint_name=args.checkpoint,
    )
