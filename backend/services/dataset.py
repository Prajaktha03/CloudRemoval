from collections import Counter
from pathlib import Path


def dataset_info(root: Path) -> dict:
    if not root.is_dir():
        return {
            "configured": False,
            "path": str(root),
            "file_count": 0,
            "files_by_extension": {},
            "note": "Dataset directory is not present. Configure DATASET_ROOT; no samples are fabricated.",
        }
    extensions = Counter(path.suffix.lower() or "[no extension]" for path in root.rglob("*") if path.is_file())
    return {
        "configured": True,
        "path": str(root),
        "file_count": sum(extensions.values()),
        "files_by_extension": dict(sorted(extensions.items())),
        "note": "Inventory only; dataset pairing, alignment, and sample metadata require a configured SEN12MS-CR-TS adapter.",
    }
