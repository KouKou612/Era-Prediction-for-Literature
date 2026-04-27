from pathlib import Path
import sys
import re
import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_utils import load_dataset
from logging_utils import start_logging
from evaluation import get_metrics, evaluate_model
from train_common import TRAINING_DIR
from config import (
    ERA_CSV,
    ERA_TEXT_DIR,
    RANDOM_STATE,
    MODEL_OUTPUT_DIR,
)


TEXT_WORD_LIMIT = 10000
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 1000
EMBED_DIM = 128
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.3

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def make_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def build_vocab(texts: list[str], max_vocab_size: int) -> dict[str, int]:
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, _ in counter.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_seq_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_seq_len]]

    if len(ids) < max_seq_len:
        ids += [vocab[PAD_TOKEN]] * (max_seq_len - len(ids))

    return ids


class BookDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        vocab: dict[str, int],
        label2id: dict[str, int],
        max_seq_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        input_ids = encode_text(self.texts[idx], self.vocab, self.max_seq_len)
        label = self.label2id[self.labels[idx]]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        dropout: float,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            pooled = torch.max(c, dim=2).values
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(model, dataloader, device, id2label: dict[int, str]) -> tuple[list[str], list[str]]:
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            y_true.extend([id2label[int(x)] for x in labels.cpu().numpy()])
            y_pred.extend([id2label[int(x)] for x in preds.cpu().numpy()])

    return y_true, y_pred


def save_checkpoint(
    model,
    vocab,
    label2id,
    id2label,
    metrics,
    epoch: int,
    run_name: str,
) -> None:
    save_dir = MODEL_OUTPUT_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    '''
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "label2id": label2id,
            "id2label": id2label,
            "config": {
                "max_seq_len": MAX_SEQ_LEN,
                "embed_dim": EMBED_DIM,
                "num_filters": NUM_FILTERS,
                "kernel_sizes": KERNEL_SIZES,
                "dropout": DROPOUT,
            },
            "best_metrics": metrics,
            "epoch": epoch,
        },
        save_dir / "best_model.pt",
    )

    print(f"Saved best model to: {save_dir / 'best_model.pt'}")
    '''

def train_one_task(csv_path: Path, text_dir: Path, label_col: str, run_name: str) -> dict:
    print(f"\n=== {run_name.upper()} ===")

    df = load_dataset(csv_path, text_dir, label_col, max_words=TEXT_WORD_LIMIT)
    print(f"Usable samples: {len(df)}")
    print(df[label_col].value_counts().sort_index())

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )

    label2id, id2label = make_label_maps(df[label_col].tolist())

    vocab = build_vocab(train_df["text"].tolist(), MAX_VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")

    train_dataset = BookDataset(
        texts=train_df["text"].tolist(),
        labels=train_df[label_col].tolist(),
        vocab=vocab,
        label2id=label2id,
        max_seq_len=MAX_SEQ_LEN,
    )

    test_dataset = BookDataset(
        texts=test_df["text"].tolist(),
        labels=test_df[label_col].tolist(),
        vocab=vocab,
        label2id=label2id,
        max_seq_len=MAX_SEQ_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=len(label2id),
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT,
        padding_idx=0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_metrics = {}
    best_state_dict = None
    best_epoch = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        y_true, y_pred = predict(model, test_loader, device, id2label)
        metrics = get_metrics(y_true, y_pred, label_col)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation macro F1: {metrics['macro_f1']:.4f}")
        print(f"Validation weighted F1: {metrics['weighted_f1']:.4f}")

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = metrics
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            save_checkpoint(
                model=model,
                vocab=vocab,
                label2id=label2id,
                id2label=id2label,
                metrics=best_metrics,
                epoch=epoch,
                run_name=run_name,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\nFinal evaluation using BEST model from epoch {best_epoch}:")
    y_true, y_pred = predict(model, test_loader, device, id2label)
    evaluate_model(y_true, y_pred, label_col)

    print("\nBest metrics:")
    print(best_metrics)

    return best_metrics


def main() -> None:
    set_seed(RANDOM_STATE)
    start_logging("train_cnn", log_dir=TRAINING_DIR)

    train_one_task(ERA_CSV, ERA_TEXT_DIR, "era", "cnn_era")


if __name__ == "__main__":
    main()