"""Checkpoint management for satellite cloud removal models."""

import os
from pathlib import Path
from typing import Any
import torch


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> Path:
    """Save training checkpoint to disk."""
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename
    temp_path = out_dir / f"{filename}.tmp"

    torch.save(state, temp_path)
    if file_path.exists():
        file_path.unlink()
    temp_path.rename(file_path)

    if is_best:
        best_path = out_dir / "best_model.pt"
        torch.save(state, best_path)

    return file_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load checkpoint with device mapping and compatibility handling."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    try:
        checkpoint = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(p, map_location=device)

    return checkpoint
