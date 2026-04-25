"""
app.py — Gradio demo for MatSciBERT Scientific NER.
Deploy to HuggingFace Spaces as the main app file.

Run locally:
    python app.py
"""

import gradio as gr
from src.inference import (
    load_pipeline, predict, to_highlighted_text,
    ENTITY_COLORS, ENTITY_DESCRIPTIONS,
)

ner_pipeline = load_pipeline()

EXAMPLES = [
    "Titanium exhibits a tensile strength of 950 MPa after sintering at 1200 °C.",
    "XRD characterisation confirmed phase purity of the synthesised alumina powder.",
    "CVD deposition produces thin diamond films used in cutting tool applications.",
    "Inconel 718 retains tensile strength beyond 700 °C due to gamma-prime precipitates.",
    "SEM imaging revealed a porosity of 2.3% in the hot-pressed silicon carbide samples.",
    "Electroplating applied a 25 µm nickel coating for corrosion protection.",
]


def run_ner(text: str):
    if not text.strip():
        return [], "Please enter a sentence."
    entities = predict(text, ner_pipeline)
    highlighted = to_highlighted_text(text, entities)
    summary = "\n\n".join(
        f"**{e.label}** · `{e.text}` · confidence {e.score:.1%}"
        for e in entities
    ) or "No entities detected."
    return highlighted, summary


with gr.Blocks(
    title="MatSciBERT Scientific NER",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 860px !important; margin: auto; }",
) as demo:

    gr.Markdown("""
# 🔬 MatSciBERT — Scientific NER

Fine-tuned **[m3rg-iitd/matscibert](https://huggingface.co/m3rg-iitd/matscibert)** on the
**[MatSci-NLP](https://huggingface.co/datasets/m3rg-iitd/MatSci-NLP)** dataset for Named
Entity Recognition across materials science literature.

Built with 🤗 **HuggingFace Transformers** · **Trainer API** · **seqeval** evaluation
""")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input sentence",
                placeholder="e.g. Sintering at 1200°C densifies the alumina powder.",
                lines=3,
            )
            run_btn = gr.Button("Extract Entities ▶", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Entity Types (MatSci-NLP):**")
            for desc in ENTITY_DESCRIPTIONS.values():
                gr.Markdown(desc)

    highlighted_output = gr.HighlightedText(
        label="Annotated text",
        combine_adjacent=True,
        color_map=ENTITY_COLORS,
    )
    summary_output = gr.Markdown(label="Extracted entities")

    gr.Examples(examples=[[s] for s in EXAMPLES], inputs=text_input)

    run_btn.click(fn=run_ner, inputs=text_input, outputs=[highlighted_output, summary_output])
    text_input.submit(fn=run_ner, inputs=text_input, outputs=[highlighted_output, summary_output])

    gr.Markdown("""
---
**Model:** MatSciBERT fine-tuned on MatSci-NLP NER  
**Source:** [GitHub](https://github.com/your-username/matscibert-ner)
""")

if __name__ == "__main__":
    demo.launch()
