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
    "Nitric oxide reacts with oxygen to form nitrogen dioxide.",
    "Mercury and its compounds are highly toxic to living organisms.",
    "Aspirin (acetylsalicylic acid) is widely used as an analgesic and anti-inflammatory.",
    "The synthesis of titanocene dichloride was confirmed by NMR spectroscopy.",
    "Ethanol is produced by fermentation of sugars by yeast.",
    "Platinum catalysts are used in the hydrogenation of organic compounds.",
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
# 🔬 MatSciBERT — Chemical NER

Fine-tuned **[m3rg-iitd/matscibert](https://huggingface.co/m3rg-iitd/matscibert)** on the
**[CHEMDNER](https://huggingface.co/datasets/kjappelbaum/chemnlp-chemdner)** dataset for Named
Entity Recognition of chemical compounds in biomedical literature.

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
            gr.Markdown("**Entity Types (CHEMDNER):**")
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
**Model:** MatSciBERT fine-tuned on CHEMDNER (chemical NER)  
**Source:** [GitHub](https://github.com/teman67/Fine-tuning-Materials-Scientific-NER-)
""")

if __name__ == "__main__":
    demo.launch()
