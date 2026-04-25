"""
dataset.py — Loads and prepares the CHEMDNER chemistry NER dataset.

Source: https://huggingface.co/datasets/kjappelbaum/chemnlp-chemdner
Paper:  "CHEMDNER: The drugs and chemical names extraction challenge" (2015)

The dataset contains chemical entity annotations from PubMed abstracts and
patents. Raw format uses character-offset spans; this module converts them to
BIO-tagged token sequences expected by the training pipeline.

NER entity types:
    O      — Outside
    B-CHEM — Beginning of a chemical entity
    I-CHEM — Inside (continuation of) a chemical entity
"""

import logging
import re
from datasets import load_dataset, DatasetDict, Dataset, Features, Sequence, Value, ClassLabel

logger = logging.getLogger(__name__)

DATASET_ID = "kjappelbaum/chemnlp-chemdner"

# BIO label schema
LABEL_NAMES = ["O", "B-CHEM", "I-CHEM"]


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Whitespace-split text, returning tokens and their (start, end) char offsets."""
    tokens = []
    offsets = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def _spans_to_bio(text: str, entities: list) -> tuple[list[str], list[int]]:
    """
    Convert entity strings to BIO token tags by finding each entity in the text.

    Each element of `entities` is a string (the entity surface form).
    Returns (tokens, ner_tag_ids).
    """
    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    tokens, offsets = _tokenize_with_offsets(text)
    if not tokens:
        return [], []

    # Find character spans for each entity string occurrence in the text
    entity_spans: list[tuple[int, int]] = []
    for ent in (entities or []):
        if not isinstance(ent, str) or not ent:
            continue
        for m in re.finditer(re.escape(ent), text):
            entity_spans.append((m.start(), m.end()))

    tag_ids = []
    for tok_start, tok_end in offsets:
        tag = "O"
        for ent_start, ent_end in entity_spans:
            if tok_start >= ent_start and tok_end <= ent_end:
                tag = "B-CHEM" if tok_start == ent_start else "I-CHEM"
                break
            elif tok_start >= ent_start and tok_start < ent_end:
                tag = "I-CHEM"
                break
        tag_ids.append(label2id[tag])

    return tokens, tag_ids


def _convert_split(hf_split) -> dict:
    """Convert raw span-annotated rows to BIO token rows."""
    all_tokens, all_tags = [], []
    for row in hf_split:
        tokens, tags = _spans_to_bio(row["text"], row["entities"])
        if tokens:
            all_tokens.append(tokens)
            all_tags.append(tags)
    return {"tokens": all_tokens, "ner_tags": all_tags}


def build_dataset(seed: int = 42) -> tuple[DatasetDict, list[str], dict, dict]:
    """
    Load and prepare the CHEMDNER NER dataset from the HuggingFace Hub.

    Returns:
        dataset     — DatasetDict with train / validation splits
        label_names — ordered list of BIO label strings
        label2id    — string → int mapping
        id2label    — int → string mapping
    """
    logger.info(f"Loading {DATASET_ID} from HuggingFace Hub...")

    try:
        raw = load_dataset(DATASET_ID)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {DATASET_ID}.\n"
            "Make sure you have internet access.\n"
            f"Original error: {e}"
        )

    # The dataset has a single 'train' split with a 'split' column
    # indicating 'train' / 'validation' / 'test'
    full = raw["train"]
    split_col = full["split"] if "split" in full.column_names else ["train"] * len(full)

    raw_train = full.filter(lambda x: x["split"] in ("train", "Train", "TRAIN"))
    raw_val   = full.filter(lambda x: x["split"] in ("validation", "val", "dev",
                                                       "Validation", "Val", "Dev"))
    if len(raw_val) == 0:
        # Fall back: 90/10 split
        splits = full.train_test_split(test_size=0.1, seed=seed)
        raw_train, raw_val = splits["train"], splits["test"]

    features = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABEL_NAMES)),
    })

    def _make_dataset(hf_split):
        converted = _convert_split(hf_split)
        return Dataset.from_dict(converted, features=features)

    logger.info("Converting span annotations to BIO token format...")
    dataset = DatasetDict({
        "train":      _make_dataset(raw_train),
        "validation": _make_dataset(raw_val),
    })

    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    id2label = {i: l for l, i in label2id.items()}

    logger.info(f"Labels ({len(LABEL_NAMES)}): {LABEL_NAMES}")
    logger.info(
        f"Split sizes — train: {len(dataset['train'])}, "
        f"validation: {len(dataset['validation'])}"
    )

    return dataset, LABEL_NAMES, label2id, id2label
