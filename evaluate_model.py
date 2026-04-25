"""
evaluate_model.py — Evaluate the fine-tuned MatSciBERT model.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model_path ./model_output/final
"""

import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

from src.dataset import build_dataset
from train import tokenize_and_align, make_compute_metrics


def run_eval(model_path: str = "./model_output/final"):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    dataset, label_names, _, _ = build_dataset()
    tokenised = dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    args = TrainingArguments(output_dir="/tmp/eval", report_to="none")
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tokenised["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(label_names),
    )

    metrics = trainer.evaluate()
    print("\n── Evaluation Results ─────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print("────────────────────────────────────────────────\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./model_output/final")
    args = parser.parse_args()
    run_eval(args.model_path)
