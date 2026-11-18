from dataclasses import dataclass
import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    optical_channels: int
    sar_channels: int
    model_base_channels: int
    model_uses_mask: bool
    checkpoint_path: Path
    dataset_root: Path
    database_path: Path
    max_upload_bytes: int
    max_decoded_bytes: int
    frontend_origins: list[str]
    session_cookie_secure: bool
    session_max_age_seconds: int
    training_command: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parents[2]
        origins = os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:3000,http://localhost:8501",
        )
        return cls(
            optical_channels=_int_env("OPTICAL_CHANNELS", 13),
            sar_channels=_int_env("SAR_CHANNELS", 2),
            model_base_channels=_int_env("MODEL_BASE_CHANNELS", 32),
            model_uses_mask=os.getenv("MODEL_USES_MASK", "true").lower() in {"1", "true", "yes"},
            checkpoint_path=Path(os.getenv("MODEL_CHECKPOINT", str(base_dir / "outputs" / "checkpoints" / "generator.pt"))),
            dataset_root=Path(os.getenv("DATASET_ROOT", str(base_dir / "data" / "raw"))),
            database_path=Path(os.getenv("EXPERIMENTS_DB", str(base_dir / "data" / "experiments.sqlite3"))),
            max_upload_bytes=_int_env("MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_decoded_bytes=_int_env("MAX_DECODED_BYTES", 128 * 1024 * 1024),
            frontend_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
            session_cookie_secure=os.getenv("SESSION_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"},
            session_max_age_seconds=_int_env("SESSION_MAX_AGE_SECONDS", 12 * 60 * 60),
            training_command=os.getenv("TRAIN_COMMAND") or None,
        )


settings = Settings.from_env()
