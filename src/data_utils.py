from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(
    csv_path: Path,
    text_dir: Path,
    label_col: str,
    max_words: int | None = 10000,
    prefix_chars: int | None = None,
    random_chunk_chars: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    required_cols = {"gutenberg_id", label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    texts = []
    keep_rows = []

    if random_chunk_chars is not None and prefix_chars is not None:
        raise ValueError("Use only one of prefix_chars or random_chunk_chars.")

    rng = (
        np.random.default_rng(random_state)
        if random_chunk_chars is not None
        else None
    )

    for _, row in df.iterrows():
        book_id = int(row["gutenberg_id"])
        text_path = text_dir / f"{book_id}.txt"

        if not text_path.exists():
            continue

        if random_chunk_chars is not None:
            text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            n = len(text)
            if n > random_chunk_chars:
                hi = n - random_chunk_chars + 1
                start = int(rng.integers(0, hi))
                text = text[start : start + random_chunk_chars]
        elif prefix_chars is not None:
            with text_path.open(encoding="utf-8", errors="ignore") as f:
                text = f.read(prefix_chars)
            text = text.strip()
            if not text:
                continue
        else:
            text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

        if max_words is not None and random_chunk_chars is None:
            text = " ".join(text.split()[:max_words])

        if not text:
            continue

        texts.append(text)
        keep_rows.append(row)

    if not keep_rows:
        raise ValueError(f"No usable texts found in {text_dir}")

    out_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    out_df["text"] = texts
    return out_df