from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    root = os.environ.get("TOKPARITY_CACHE") or Path.home() / ".cache" / "tokparity"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path
