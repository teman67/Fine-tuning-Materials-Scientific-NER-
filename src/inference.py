"""
inference.py — Load the fine-tuned MatSciBERT model and run NER predictions.
"""

from pathlib import Path
from typing import NamedTuple
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# MatSci-NLP entity type → display colour (for Gradio HighlightedText)
# Labels are dataset-native; colours degrade gracefully for unknown types.
ENTITY_COLORS = {
    "MAT":  "#c8e6c9",   # green  — material
    "PRO":  "#bbdefb",   # blue   — property
    "SPL":  "#e1bee7",   # purple — sample descriptor
    "SMT":  "#f8bbd0",   # pink   — synthesis method
    "CMT":  "#ffe0b2",   # orange — characterisation method
    "CHR":  "#fff9c4",   # yellow — characterisation (alt label)
    "APL":  "#b2ebf2",   # teal   — application
}

# Human-readable descriptions shown in the Gradio sidebar
ENTITY_DESCRIPTIONS = {
    "MAT": "🟢 Material      — e.g. titanium, PEEK, carbon fibre",
    "PRO": "🔵 Property      — e.g. tensile strength, melting point",
    "SPL": "🟣 Sample        — e.g. thin film, bulk sample",
    "SMT": "🔴 Synthesis     — e.g. sintering, CVD deposition",
    "CMT": "🟠 Characterisation — e.g. XRD, SEM, EBSD",
    "APL": "🩵 Application   — e.g. turbine blades, fuel cells",
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
