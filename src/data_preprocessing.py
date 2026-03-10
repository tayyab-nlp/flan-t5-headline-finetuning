"""Data loading and preprocessing for Gigaword headline training."""

from __future__ import annotations

from typing import Dict, Optional

from datasets import Dataset, load_dataset

from .utils import cleanup_text

PREFIX = "headline: "


def format_example(example: Dict[str, str]) -> Dict[str, str]:
    """Convert Gigaword row into FLAN-T5 seq2seq format."""
    source = cleanup_text(example.get("article"))
    target = cleanup_text(example.get("summary"))
    return {
        "input_text": f"{PREFIX}{source}",
        "target_text": target,
    }


def load_gigaword_splits(
    dataset_id: str = "SalmanFaroz/gigaword",
    train_samples: int = 10000,
    validation_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Load and optionally downsample Gigaword train/validation splits."""
    dataset = load_dataset(dataset_id)

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if train_samples and train_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=seed).select(range(train_samples))
    if validation_samples and validation_samples < len(val_ds):
        val_ds = val_ds.shuffle(seed=seed).select(range(validation_samples))

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

    return {"train": train_ds, "validation": val_ds}


def tokenize_batch(
    batch: Dict[str, list],
    tokenizer,
    max_input_length: int,
    max_target_length: int,
) -> Dict[str, list]:
    """Tokenize batched source/target text for sequence-to-sequence training."""
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
