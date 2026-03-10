"""General utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> str:
    """Create directory if missing and return its path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def cleanup_text(text: Optional[str]) -> str:
    """Normalize text whitespace for stable training and inference."""
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split()).strip()
