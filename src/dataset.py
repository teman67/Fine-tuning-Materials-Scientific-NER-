"""
dataset.py — Loads and prepares the MatSci-NLP NER dataset.

Source: https://huggingface.co/datasets/m3rg-iitd/MatSci-NLP
Paper:  "MatSci-NLP: Evaluating Scientific NLP on Materials Science" (2023)

MatSci-NLP NER entity types (native):
    MAT  — Material
    PRO  — Property
    CHR  — Characterisation method
    SPL  — Sample descriptor
    SMT  — Synthesis method
    CMT  — Characterisation method (alternate split)
    O    — Outside

The dataset ships with BIO-tagged token sequences and is directly loadable
from the HuggingFace Hub.
"""

import logging
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)

DATASET_ID = "m3rg-iitd/MatSci-NLP"
DATASET_CONFIG = "ner"          # NER subset of the MatSci-NLP benchmark


def build_dataset(seed: int = 42) -> tuple[DatasetDict, list[str], dict, dict]:
    """
    Load the MatSci-NLP NER dataset from the HuggingFace Hub.

    Returns:
        dataset    — DatasetDict with train / validation / test splits
        label_names — ordered list of BIO label strings
        label2id   — string → int mapping
        id2label   — int → string mapping
    """
    logger.info(f"Loading {DATASET_ID} ({DATASET_CONFIG}) from HuggingFace Hub...")

    try:
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {DATASET_ID}.\n"
            "Make sure you have internet access and run:\n"
            "  huggingface-cli login   (if the dataset requires acceptance of terms)\n"
            f"Original error: {e}"
        )

    # ------------------------------------------------------------------
    # Extract label names from the dataset's ClassLabel feature
    # ------------------------------------------------------------------
    ner_feature = dataset["train"].features["ner_tags"]

    # Handle both Sequence(ClassLabel(...)) and plain ClassLabel
    if hasattr(ner_feature, "feature"):
        label_names: list[str] = ner_feature.feature.names
    else:
        label_names = ner_feature.names

    label2id = {l: i for i, l in enumerate(label_names)}
    id2label = {i: l for l, i in label2id.items()}

    logger.info(f"Labels ({len(label_names)}): {label_names}")
    logger.info(
        f"Split sizes — train: {len(dataset['train'])}, "
        f"validation: {len(dataset.get('validation', dataset.get('test', [])))} "
    )

    # Rename 'test' → 'validation' if needed for Trainer compatibility
    if "validation" not in dataset and "test" in dataset:
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["test"],
        })

    return dataset, label_names, label2id, id2label
