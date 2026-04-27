from pathlib import Path
import numpy as np
import pandas as pd


from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
    
from data_utils import load_dataset
from logging_utils import start_logging
from evaluation import get_metrics
from train_common import TRAINING_DIR
from config import (
    ERA_CSV,
    ERA_TEXT_DIR,
    RANDOM_STATE,
    MODEL_OUTPUT_DIR,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
TEXT_WORD_LIMIT = 10000 


def make_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def build_hf_datasets(df: pd.DataFrame, label_col: str):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )

    label2id, id2label = make_label_maps(df[label_col].tolist())

    train_df = train_df[["text", label_col]].copy()
    test_df = test_df[["text", label_col]].copy()

    train_df["labels"] = train_df[label_col].map(label2id)
    test_df["labels"] = test_df[label_col].map(label2id)

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False)

    return train_ds, test_ds, label2id, id2label


def tokenize_datasets(train_ds, test_ds, tokenizer):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)
    return train_ds, test_ds


def train_one_task(csv_path: Path, text_dir: Path, label_col: str, run_name: str):
    print(f"\n=== {run_name.upper()} ===")

    df = load_dataset(csv_path, text_dir, label_col, max_words=TEXT_WORD_LIMIT)
    print(f"Usable samples: {len(df)}")
    print(df[label_col].value_counts().sort_index())

    train_ds, test_ds, label2id, id2label = build_hf_datasets(df, label_col)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    train_ds, test_ds = tokenize_datasets(train_ds, test_ds, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        y_true = [id2label[int(x)] for x in labels]
        y_pred = [id2label[int(x)] for x in preds]

        metrics = get_metrics(y_true, y_pred, label_col)
        return metrics

    args = TrainingArguments(
        output_dir=str(Path("outputs") / run_name),
        eval_strategy="epoch",
        save_strategy="no", 
        logging_strategy="epoch",
        load_best_model_at_end=False, 
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=RANDOM_STATE,
        report_to="none",
    )   

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # trainer.save_model(str(MODEL_OUTPUT_DIR))
    # tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    # print(f"Saved model to: {MODEL_OUTPUT_DIR}")


def main():
    start_logging("train_bert", log_dir=TRAINING_DIR)

    train_one_task(ERA_CSV, ERA_TEXT_DIR, "era", "bert_era")


if __name__ == "__main__":
    main()