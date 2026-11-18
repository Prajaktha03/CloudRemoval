"""Inference pipeline for Multimodal Satellite Cloud Removal."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.cloud_masks import detect_s2_clouds
from datasets.preprocessing import normalize_optical, normalize_sar
from evaluation.metrics import evaluate_sample_metrics
from inference.geotiff_io import read_satellite_image, write_satellite_image
from models.generator import MultimodalUNet
from models.unet_baseline import OpticalOnlyUNet
from training.checkpoint import load_checkpoint
from training.losses import composite_cloud_free


@dataclass
class PredictionOutput:
    reconstruction: np.ndarray
    cloud_mask: np.ndarray
    optical_input: np.ndarray
    sar_input: np.ndarray
    reference: np.ndarray | None
    metrics: dict[str, Any] | None
    metadata: dict[str, Any]


class CloudRemovalPredictor:
    """Inference engine for multimodal satellite cloud removal."""

    def __init__(self, checkpoint_path: str | Path, device_str: str = "auto") -> None:
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.config = {}
        self.load()

    def load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {self.checkpoint_path}")

        checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)
        self.config = checkpoint.get("config", {})

        opt_ch = self.config.get("data", {}).get("optical_channels", 13)
        sar_ch = self.config.get("data", {}).get("sar_channels", 2)
        use_mask = self.config.get("model", {}).get("use_mask_input", True)
        base_ch = self.config.get("model", {}).get("base_channels", 32)
        model_type = checkpoint.get("model_type", "multimodal_cgan")

        if model_type == "optical_only_unet":
            self.model = OpticalOnlyUNet(
                optical_channels=opt_ch,
                use_mask=use_mask,
                base_channels=base_ch,
            ).to(self.device)
        else:
            self.model = MultimodalUNet(
                optical_channels=opt_ch,
                sar_channels=sar_ch,
                use_mask=use_mask,
                base_channels=base_ch,
            ).to(self.device)

        state = checkpoint.get("generator", checkpoint.get("model_state_dict", checkpoint))
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(
        self,
        cloudy_optical: np.ndarray,
        sar: np.ndarray,
        cloud_mask: np.ndarray | None = None,
        reference: np.ndarray | None = None,
        use_compositing: bool = True,
    ) -> PredictionOutput:
        """Run cloud removal reconstruction on multimodal input arrays.
        
        Args:
            cloudy_optical: Normalized optical array in range [0, 1] with shape (H, W, C).
            sar: Normalized SAR array in range [0, 1] with shape (H, W, 2).
            cloud_mask: Optional binary mask (H, W, 1). If None, detected automatically.
            reference: Optional clear reference optical array for quality assessment.
            use_compositing: Whether to blend model reconstruction in cloud areas with original clear pixels.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Ensure shapes
        if cloudy_optical.ndim == 2:
            cloudy_optical = cloudy_optical[..., np.newaxis]
        if sar.ndim == 2:
            sar = sar[..., np.newaxis]

        # Automatic cloud detection if mask not supplied
        if cloud_mask is None:
            cloud_mask = detect_s2_clouds(cloudy_optical)

        # Convert to PyTorch tensors (1, C, H, W)
        opt_t = torch.from_numpy(np.moveaxis(cloudy_optical, -1, 0)).unsqueeze(0).float().to(self.device)
        sar_t = torch.from_numpy(np.moveaxis(sar, -1, 0)).unsqueeze(0).float().to(self.device)
        mask_t = torch.from_numpy(np.moveaxis(cloud_mask, -1, 0)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            if isinstance(self.model, OpticalOnlyUNet):
                pred_t = self.model(opt_t, mask_t)
            else:
                pred_t = self.model(opt_t, sar_t, mask_t)

            if use_compositing:
                pred_t = composite_cloud_free(pred_t, opt_t, mask_t)

        reconstruction = np.moveaxis(pred_t[0].cpu().numpy(), 0, -1)
        reconstruction = np.clip(reconstruction, 0.0, 1.0)

        # Compute evaluation metrics if ground-truth reference is available
        metrics = None
        if reference is not None:
            if reference.shape == reconstruction.shape:
                metrics = evaluate_sample_metrics(reconstruction, reference, cloud_mask)

        metadata = {
            "checkpoint": str(self.checkpoint_path.resolve()),
            "device": str(self.device),
            "compositing_enabled": use_compositing,
            "optical_shape": list(cloudy_optical.shape),
            "sar_shape": list(sar.shape),
            "cloud_fraction": float(np.mean(cloud_mask > 0.5)),
        }

        return PredictionOutput(
            reconstruction=reconstruction,
            cloud_mask=cloud_mask,
            optical_input=cloudy_optical,
            sar_input=sar,
            reference=reference,
            metrics=metrics,
            metadata=metadata,
        )
