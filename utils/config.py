from pathlib import Path
from typing import Any
import yaml


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot access."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_yaml_config(path: str | Path) -> ConfigDict:
    """Load configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ConfigDict(data)


def save_yaml_config(config: dict, path: str | Path) -> None:
    """Save configuration to a YAML file."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, default_flow_style=False, sort_keys=False)
