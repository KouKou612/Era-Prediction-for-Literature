"""DistilBERT fine-tuning for era and decade classification."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    ERA_CSV, ERA_TEXT_DIR,
    DECADE_CSV, DECADE_TEXT_DIR,
    RANDOM_STATE, TRAIN_CONFIG, TRANSFORMER_CONFIG,
)
from data_utils import load_dataset
from evaluation import evaluate_model, get_metrics
from logging_utils import start_logging
from train_common import TRAINING_DIR


def _load_data(csv_path, text_dir, label_col):
    random_chunk = TRAIN_CONFIG.get("random_chunk_chars")
    chunk_seed = TRAIN_CONFIG.get("chunk_random_state", RANDOM_STATE)
    if random_chunk is not None:
        return load_dataset(csv_path, text_dir, label_col,
                            max_words=None, random_chunk_chars=random_chunk, random_state=chunk_seed)
    return load_dataset(csv_path, text_dir, label_col, max_words=TRAIN_CONFIG["max_words"])


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def train_transformer_for_task(*, csv_path, text_dir, label_col, output_dir):
    df = _load_data(csv_path, text_dir, label_col)

    label_list = sorted(df[label_col].unique().tolist())
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    df = df.copy()
    df["label"] = df[label_col].map(label2id).astype(int)

    train_df, test_df = train_test_split(
        df, test_size=TRAIN_CONFIG["test_size"],
        random_state=RANDOM_STATE, stratify=df["label"],
    )

    checkpoint = TRANSFORMER_CONFIG["checkpoint"]
    max_length = int(TRANSFORMER_CONFIG["max_length"])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(label_list), label2id=label2id, id2label=id2label,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            seed=RANDOM_STATE,
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_true_lbl = [id2label[int(i)] for i in test_df["label"].to_numpy()]
    y_pred_lbl = [id2label[int(i)] for i in np.argmax(preds.predictions, axis=-1)]

    print(f"\n=== Transformer | {label_col.upper()} ===")
    print(f"Checkpoint: {checkpoint}  Max length: {max_length}")
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    evaluate_model(y_true_lbl, y_pred_lbl, label_col)

    metrics = get_metrics(y_true_lbl, y_pred_lbl, label_col)
    metrics["n_train"] = len(train_df)
    metrics["n_test"] = len(test_df)
    print("\nMetrics:", metrics)
    return metrics


def main():
    start_logging("train_transformer", log_dir=TRAINING_DIR)

    era_metrics = train_transformer_for_task(
        csv_path=ERA_CSV, text_dir=ERA_TEXT_DIR,
        label_col="era", output_dir="runs/transformer_era",
    )
    print("\n" + "=" * 80 + "\n")

    decade_metrics = train_transformer_for_task(
        csv_path=DECADE_CSV, text_dir=DECADE_TEXT_DIR,
        label_col="decade", output_dir="runs/transformer_decade",
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nERA:", era_metrics)
    print("\nDECADE:", decade_metrics)


if __name__ == "__main__":
    main()
