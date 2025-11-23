import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict


ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clean_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
