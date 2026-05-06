from pathlib import Path
import random
import pandas as pd


DATASET_DIR = Path("Dataset")

INPUT_CSV = DATASET_DIR / "sample_by_era.csv"
TEXT_DIR = DATASET_DIR / "era_sample_clean"
OUTPUT_CSV = DATASET_DIR / "sample_by_era_with_chunks.csv"

CHUNK_SIZE = 1000
RANDOM_STATE = 612


def main():
    random.seed(RANDOM_STATE)

    df = pd.read_csv(INPUT_CSV)

    chunks = []

    for i, row in df.iterrows():
        book_id = int(row["gutenberg_id"])
        text_path = TEXT_DIR / f"{book_id}.txt"

        if not text_path.exists():
            print(f"Missing: {book_id}")
            chunks.append("")
            continue

        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            print(f"Empty: {book_id}")
            chunks.append("")
            continue

        words = text.split()
        if len(words) <= CHUNK_SIZE:
            chunk = " ".join(words)
        else:
            start = random.randint(0, len(words) - CHUNK_SIZE)
            chunk = " ".join(words[start : start + CHUNK_SIZE])
        chunks.append(chunk)

    df["chunk"] = chunks

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
