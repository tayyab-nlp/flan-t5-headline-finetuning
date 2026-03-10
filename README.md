# FLAN-T5 Fine-Tuning for News-to-Headline Generation

Space Demo Link: [https://vtayyab6-flan-t5-headline-finetuning.hf.space](https://vtayyab6-flan-t5-headline-finetuning.hf.space)

Fine-tune FLAN-T5-Small on Gigaword to convert news-style input text into concise headline outputs, then serve it through a fast Gradio interface.

## Features

| Area | Details |
|---|---|
| Task | News sentence/paragraph -> short headline generation |
| Model | `google/flan-t5-small` (uses fine-tuned checkpoint if available) |
| Dataset | Hugging Face Gigaword |
| Training | Lightweight `train.py` with configurable sample size |
| Inference | Reusable `generate_headline()` pipeline in `inference.py` |
| UI | Gradio app with input box, generate button, examples, and decoding controls |
| Deployment | Hugging Face Spaces ready |

## Project Structure

```text
flan-t5-headline-finetuning/
├── app.py
├── train.py
├── inference.py
├── requirements.txt
├── requirements-train.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_utils.py
│   └── utils.py
└── outputs/
    └── saved_model/
```

## How It Works

1. Load Gigaword (`document`, `summary`).
2. Format input as `headline: <news text>` and target as `<headline>`.
3. Fine-tune FLAN-T5-Small.
4. Save model and tokenizer to `outputs/saved_model`.
5. Run Gradio app and generate headlines from new inputs.

## Local Setup

```bash
cd flan-t5-headline-finetuning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

## Train Locally

```bash
pip install -r requirements-train.txt
python3 train.py --train_samples 10000 --validation_samples 1000 --num_train_epochs 1
```

The app automatically loads `outputs/saved_model` if present.
