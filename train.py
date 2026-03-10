"""Fine-tune FLAN-T5-small on Gigaword for headline generation."""

from __future__ import annotations

import argparse

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
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--output_dir", type=str, default="outputs/saved_model")
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--validation_samples", type=int, default=1000)
    parser.add_argument("--max_input_length", type=int, default=192)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)

    print("Loading dataset...")
    splits = load_gigaword_splits(
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
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        report_to="none",
        fp16=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
