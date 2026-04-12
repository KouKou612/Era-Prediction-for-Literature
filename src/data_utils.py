from pathlib import Path
import pandas as pd


def load_dataset(csv_path: Path, text_dir: Path, label_col: str, max_words: int = 10000) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    required_cols = {"gutenberg_id", label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    texts = []
    keep_rows = []

    for _, row in df.iterrows():
        book_id = int(row["gutenberg_id"])
        text_path = text_dir / f"{book_id}.txt"

        if not text_path.exists():
            continue

        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        if max_words is not None:
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