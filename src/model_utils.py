"""Model/tokenizer loading utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@lru_cache(maxsize=2)
def load_model_and_tokenizer(model_name_or_path: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Load and cache tokenizer/model to avoid repeated startup overhead."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.eval()
    return tokenizer, model
