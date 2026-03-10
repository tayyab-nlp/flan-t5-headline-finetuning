"""Gradio app for FLAN-T5 headline generation."""

from __future__ import annotations

import os
from typing import Tuple

import gradio as gr

from inference import generate_headline

APP_TITLE = "FLAN-T5 Fine-Tuning for News-to-Headline Generation"
APP_SUBTITLE = (
    "Convert news sentences into concise headlines using a FLAN-T5 model fine-tuned on Gigaword."
)

APP_CSS = """
#app-shell {
  max-width: 1180px;
  margin: 0 auto;
}
.panel-card {
  border: 1px solid #d8e0ef;
  border-radius: 14px;
  padding: 12px;
}
.output-card {
  border: 1px solid #e2e8f3;
  border-radius: 12px;
  padding: 12px;
}
#generate-btn {
  min-height: 44px;
}
"""

EXAMPLES = [
    [
        "The government announced a new transport reform plan on Tuesday to improve rail connectivity across the country."
    ],
    [
        "Apple introduced its latest wearable devices during an event focused on health tracking and battery performance improvements."
    ],
    [
        "Heavy rainfall disrupted flights and road traffic in several districts as emergency teams were deployed overnight."
    ],
]


def run_generation(news_text: str, max_new_tokens: int, num_beams: int) -> Tuple[str, str, str]:
    """Generate headline and return concise status metadata."""
    cleaned = (news_text or "").strip()
    if not cleaned:
        return (
            "Please enter a news sentence or paragraph.",
            "Waiting for input.",
            "",
        )

    headline, info = generate_headline(
        text=cleaned,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    status = "Done. Headline generated."
    details = "\n".join(
        [
            f"- Loaded model: `{info['loaded_model']}`",
            f"- Device: `{info['device']}`",
            f"- Max new tokens: `{max_new_tokens}`",
            f"- Beams: `{num_beams}`",
        ]
    )

    return headline, status, details


with gr.Blocks(title=APP_TITLE) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_SUBTITLE)

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["panel-card"]):
                with gr.Tabs():
                    with gr.Tab("Task"):
                        news_input = gr.Textbox(
                            label="News Input",
                            lines=8,
                            placeholder="Paste a news sentence or short paragraph...",
                        )
                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=[news_input],
                            label="Sample Inputs",
                        )

                    with gr.Tab("Generation Settings"):
                        max_tokens = gr.Slider(
                            minimum=8,
                            maximum=32,
                            value=20,
                            step=1,
                            label="Max New Tokens",
                        )
                        num_beams = gr.Slider(
                            minimum=1,
                            maximum=6,
                            value=4,
                            step=1,
                            label="Beam Size",
                        )
                        gr.Markdown("Model was designed for Gigaword-style headline generation.")

                with gr.Row():
                    generate_btn = gr.Button("Generate Headline", variant="primary", elem_id="generate-btn")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=6, elem_classes=["panel-card"]):
                gr.Markdown("## Output")
                status_output = gr.Markdown("Ready.")
                with gr.Tabs():
                    with gr.Tab("Headline"):
                        headline_output = gr.Textbox(
                            label="Generated Headline",
                            lines=2,
                            interactive=False,
                        )
                    with gr.Tab("Run Details"):
                        details_output = gr.Markdown("", elem_classes=["output-card"])

        generate_btn.click(
            fn=run_generation,
            inputs=[news_input, max_tokens, num_beams],
            outputs=[headline_output, status_output, details_output],
        )

        clear_btn.click(
            fn=lambda: ("", "", "Ready.", ""),
            inputs=[],
            outputs=[news_input, headline_output, status_output, details_output],
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(default_concurrency_limit=2, max_size=24).launch(
        server_name="0.0.0.0",
        server_port=port,
        css=APP_CSS,
    )
