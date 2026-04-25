"""
train.py — Fine-tune MatSciBERT on the CHEMDNER chemical NER dataset.

Uses the HuggingFace Trainer API with:
  - Token classification head on m3rg-iitd/matscibert
  - CHEMDNER dataset (kjappelbaum/chemnlp-chemdner) from HuggingFace Hub
  - seqeval metrics (precision, recall, F1)
  - Model saved locally + optionally pushed to the HF Hub

Usage:
    python train.py
    python train.py --push_to_hub --hub_model_id your-username/matscibert-ner
"""

import argparse
import logging
import numpy as np

import evaluate
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.dataset import build_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_MODEL = "m3rg-iitd/matscibert"   # domain-specific BERT for materials science
OUTPUT_DIR = "./model_output"
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LENGTH = 128


# ------------------------------------------------------------------
# Tokenisation with label alignment
# ------------------------------------------------------------------

def tokenize_and_align(examples, tokenizer):
    """
    Tokenise word-level inputs and align NER tags to sub-word tokens.
    Sets label = -100 for special tokens and continuation sub-words
    so they are ignored by the loss.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)             # [CLS] / [SEP]
            elif word_id != prev_word:
                aligned.append(labels[word_id])  # first sub-token → real label
            else:
                aligned.append(-100)             # continuation sub-token → ignore
            prev_word = word_id
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized


# ------------------------------------------------------------------
# seqeval metric computation
# ------------------------------------------------------------------

def make_compute_metrics(label_names):
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        true_preds = [
            [label_names[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for l in label_row if l != -100]
            for label_row in labels
        ]

        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": round(results["overall_precision"], 4),
            "recall":    round(results["overall_recall"], 4),
            "f1":        round(results["overall_f1"], 4),
            "accuracy":  round(results["overall_accuracy"], 4),
        }

    return compute_metrics


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def train(push_to_hub: bool = False, hub_model_id: str = None):
    logger.info("Loading CHEMDNER dataset from HuggingFace Hub...")
    dataset, label_names, label2id, id2label = build_dataset()

    logger.info(f"Loading tokenizer & model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    logger.info("Tokenising dataset...")
    tokenised = dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_names),
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating best model...")
    metrics = trainer.evaluate()
    logger.info(f"Final metrics: {metrics}")

    logger.info(f"Saving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {hub_model_id}")
        trainer.push_to_hub()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="e.g. your-username/matscibert-ner")
    args = parser.parse_args()
    train(push_to_hub=args.push_to_hub, hub_model_id=args.hub_model_id)
