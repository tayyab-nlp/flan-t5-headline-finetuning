"""Reusable inference pipeline for headline generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from src.model_utils import load_model_and_tokenizer
from src.utils import cleanup_text

DEFAULT_BASE_MODEL = "google/flan-t5-small"
LOCAL_MODEL_PATH = Path("outputs/saved_model")


class HeadlineGenerator:
    """Load model once and provide fast headline generation."""

    def __init__(self, model_name_or_path: str | None = None):
        resolved = self._resolve_model_id(model_name_or_path)
        self.model_id = resolved
        self.tokenizer, self.model = load_model_and_tokenizer(resolved)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @staticmethod
    def _resolve_model_id(model_name_or_path: str | None) -> str:
        """Resolve precedence: explicit arg -> env -> local checkpoint -> base model."""
        if model_name_or_path:
            return model_name_or_path

        env_model = os.getenv("HEADLINE_MODEL_ID", "").strip()
        if env_model:
            return env_model

        if (LOCAL_MODEL_PATH / "config.json").exists():
            return str(LOCAL_MODEL_PATH)

        return DEFAULT_BASE_MODEL

    def generate_headline(
        self,
        news_text: str,
        max_new_tokens: int = 20,
        num_beams: int = 4,
    ) -> str:
        """Generate concise headline from input news text."""
        cleaned = cleanup_text(news_text)
        if not cleaned:
            return "Please enter a news sentence or short paragraph."

        prompt = f"headline: {cleaned}"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=192,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                length_penalty=0.9,
                early_stopping=True,
            )

        headline = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return cleanup_text(headline)

    def info(self) -> Dict[str, str]:
        """Return model metadata for UI display."""
        return {
            "loaded_model": self.model_id,
            "device": self.device,
            "base_fallback": DEFAULT_BASE_MODEL,
        }


_GENERATOR: HeadlineGenerator | None = None


def get_generator() -> HeadlineGenerator:
    """Get singleton generator for the app."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = HeadlineGenerator()
    return _GENERATOR


def generate_headline(text: str, max_new_tokens: int = 20, num_beams: int = 4) -> Tuple[str, Dict[str, str]]:
    """Module-level inference helper used by app.py."""
    generator = get_generator()
    headline = generator.generate_headline(
        news_text=text,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return headline, generator.info()
