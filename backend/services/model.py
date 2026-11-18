from typing import Any
import pickle

import numpy as np
import torch

from backend.core.config import Settings
from backend.core.errors import InputValidationError, ModelUnavailableError
from models.generator import MultimodalUNet


class ModelService:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MultimodalUNet | None = None
        self.load_error: str | None = None

    def load(self) -> None:
        checkpoint = self.config.checkpoint_path
        if not checkpoint.is_file():
            self.load_error = f"Checkpoint not found: {checkpoint}"
            return
        model = MultimodalUNet(
            optical_channels=self.config.optical_channels,
            sar_channels=self.config.sar_channels,
            use_mask=self.config.model_uses_mask,
            base_channels=self.config.model_base_channels,
        )
        try:
            # Training checkpoints may contain ConfigDict metadata. Map that
            # class to a plain dict so weights-only loading stays safe.
            with torch.serialization.safe_globals([(dict, "utils.config.ConfigDict")]):
                state: Any = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if isinstance(state, dict):
                state = state.get("generator", state.get("model_state_dict", state))
            model.load_state_dict(state)
            model.to(self.device).eval()
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
            self.load_error = f"Unable to load checkpoint: {exc}"
            return
        self.model = model
        self.load_error = None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.model is not None,
            "checkpoint": str(self.config.checkpoint_path),
            "device": str(self.device),
            "optical_channels": self.config.optical_channels,
            "sar_channels": self.config.sar_channels,
            "uses_cloud_mask": self.config.model_uses_mask,
            "error": self.load_error,
        }

    def predict(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        cloud_mask: np.ndarray | None,
    ) -> np.ndarray:
        if self.model is None:
            raise ModelUnavailableError(self.load_error or "Trained model is unavailable")
        expected_mask = self.config.model_uses_mask
        if expected_mask and cloud_mask is None:
            cloud_mask = np.zeros((*optical.shape[:2], 1), dtype=np.float32)
        if not expected_mask and cloud_mask is not None:
            cloud_mask = None
        inputs = [optical, sar]
        if expected_mask:
            assert cloud_mask is not None
            inputs.append(cloud_mask)
        tensors = [
            torch.from_numpy(np.moveaxis(value, -1, 0)).unsqueeze(0).to(self.device)
            for value in inputs
        ]
        if tensors[0].shape[-2] < 16 or tensors[0].shape[-1] < 16:
            raise InputValidationError("Input dimensions must be at least 16 by 16 pixels")
        with torch.inference_mode():
            output = self.model(tensors[0], tensors[1], tensors[2] if len(tensors) == 3 else None)
        result = np.moveaxis(output[0].cpu().numpy(), 0, -1)
        return result.astype(np.float32, copy=False)
