# FLAN-T5 Fine-Tuning for News-to-Headline Generation

Space Demo Link: [https://vtayyab6-flan-t5-headline-finetuning.hf.space](https://vtayyab6-flan-t5-headline-finetuning.hf.space)

Model Link: [https://huggingface.co/vtayyab6/flan-t5-small-gigaword-headline-ft](https://huggingface.co/vtayyab6/flan-t5-small-gigaword-headline-ft)

A lightweight NLP project that turns news-style text into clean headline outputs using FLAN-T5 fine-tuned on Gigaword, with a fast Gradio demo for real-time testing.

## Features

| Area | Details |
|---|---|
| Task | News sentence/paragraph -> concise headline |
| Dataset | `SalmanFaroz/gigaword` (`article` -> `summary`) |
| Fine-tuning | `train.py` with configurable subsampling for quick runs |
| Inference | Reusable `generate_headline()` function in `inference.py` |
| UI | Gradio interface with sample inputs and generation controls |
| Deployment | Live Hugging Face Space + published fine-tuned model |
| Saved Results | `outputs/training_summary.json` and `outputs/sample_predictions.json` |

## Screenshots

Before running:

![Before run](screenshots/1.%20before%20run.png)

Generation settings:

![Generation settings](screenshots/2.%20generation%20settings.png)

Sample headline generation:

![Sample generation](screenshots/3.%20sample%20generation.png)

Run details (model/device/decoding):

![Run details](screenshots/4.%20run%20details.png)

## Project Structure

```text
flan-t5-headline-finetuning/
├── app.py
├── inference.py
├── train.py
├── requirements.txt
├── requirements-train.txt
├── README.md
├── .gitignore
├── src/
│   ├── data_preprocessing.py
│   ├── model_utils.py
│   └── utils.py
├── outputs/
│   ├── training_summary.json
│   ├── sample_predictions.json
│   └── saved_model/
└── screenshots/
```

## How It Works

1. Load Gigaword and format each row as `headline: <article>` -> `<summary>`.
2. Fine-tune `google/flan-t5-small` on a subsample for fast iteration.
3. Save model and tokenizer, then publish the checkpoint on Hugging Face.
4. Use the Gradio app to generate headlines from fresh input text.
5. Review run details directly in the UI.

## Local Setup

```bash
cd flan-t5-headline-finetuning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Quick training run (subsample):

```bash
pip install -r requirements-train.txt
python3 train.py --dataset_id SalmanFaroz/gigaword --train_samples 4000 --validation_samples 300 --max_steps 70
```
