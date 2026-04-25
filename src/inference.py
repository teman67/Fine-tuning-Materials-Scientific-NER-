"""
inference.py — Load the fine-tuned MatSciBERT model and run NER predictions.
"""

from pathlib import Path
from typing import NamedTuple
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# CHEMDNER entity type → display colour (for Gradio HighlightedText)
ENTITY_COLORS = {
    "CHEM": "#c8e6c9",   # green — chemical entity
}

# Human-readable descriptions shown in the Gradio sidebar
ENTITY_DESCRIPTIONS = {
    "CHEM": "🟢 Chemical — e.g. nitric oxide, ethanol, NaCl, aspirin",
}


class Entity(NamedTuple):
    text: str
    label: str
    start: int
    end: int
    score: float


def load_pipeline(model_path: str = "./model_output/final"):
    """
    Load the fine-tuned NER pipeline.
    Falls back to base MatSciBERT (untrained head) if model not yet trained.
    """
    path = Path(model_path)
    if not path.exists():
        print(
            f"[inference] Model not found at '{model_path}'. "
            "Run train.py first. Using base MatSciBERT in the meantime."
        )
        model_id = "m3rg-iitd/matscibert"
    else:
        model_id = str(path)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def predict(text: str, ner_pipeline) -> list[Entity]:
    raw = ner_pipeline(text)
    return [
        Entity(
            text=item["word"],
            label=item["entity_group"].replace("B-", "").replace("I-", ""),
            start=item["start"],
            end=item["end"],
            score=round(item["score"], 3),
        )
        for item in raw
    ]


def to_highlighted_text(text: str, entities: list[Entity]) -> list[tuple]:
    """Convert entity spans to Gradio HighlightedText format."""
    result = []
    cursor = 0
    for ent in sorted(entities, key=lambda e: e.start):
        if ent.start > cursor:
            result.append((text[cursor:ent.start], None))
        result.append((text[ent.start:ent.end], ent.label))
        cursor = ent.end
    if cursor < len(text):
        result.append((text[cursor:], None))
    return result
