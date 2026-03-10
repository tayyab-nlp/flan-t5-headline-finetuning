"""Fine-tune FLAN-T5-small on Gigaword for headline generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data_preprocessing import load_gigaword_splits, tokenize_batch
from src.model_utils import load_model_and_tokenizer
from src.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 on Gigaword")
    parser.add_argument("--dataset_id", type=str, default="SalmanFaroz/gigaword")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--output_dir", type=str, default="outputs/saved_model")
    parser.add_argument("--results_dir", type=str, default="outputs")
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--validation_samples", type=int, default=1000)
    parser.add_argument("--max_input_length", type=int, default=192)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.results_dir)

    print("Loading dataset...")
    splits = load_gigaword_splits(
        dataset_id=args.dataset_id,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        seed=args.seed,
    )

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    print("Tokenizing data...")
    tokenized_train = splits["train"].map(
        lambda x: tokenize_batch(x, tokenizer, args.max_input_length, args.max_target_length),
        batched=True,
    )
    tokenized_val = splits["validation"].map(
        lambda x: tokenize_batch(x, tokenizer, args.max_input_length, args.max_target_length),
        batched=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=2,
        save_only_model=True,
        predict_with_generate=True,
        report_to="none",
        fp16=False,
        eval_strategy="no",
        save_strategy="no",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    train_output = trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save concise training summary artifact for GitHub tracking.
    metrics = {
        "dataset_id": args.dataset_id,
        "model_name": args.model_name,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "train_runtime_sec": train_output.metrics.get("train_runtime"),
        "train_loss": train_output.metrics.get("train_loss"),
        "global_steps": int(train_output.global_step),
        "output_dir": args.output_dir,
    }
    summary_path = Path(args.results_dir) / "training_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save a few sample predictions to show fine-tuned behavior.
    sample_ds = splits["validation"].select(range(min(20, len(splits["validation"]))))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prediction_rows = []
    for row in sample_ds:
        encoded = tokenizer(
            row["input_text"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_target_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        prediction_rows.append(
            {
                "input_text": row["input_text"],
                "target_text": row["target_text"],
                "prediction": prediction,
            }
        )
    preds_path = Path(args.results_dir) / "sample_predictions.json"
    preds_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")

    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
