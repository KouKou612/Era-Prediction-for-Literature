from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BOOK_LIST = ROOT / "Dataset" / "selected_books.csv"
CLEAN_DIR = ROOT / "Dataset" / "books_clean"

BOOK_OUT = ROOT / "Dataset" / "book_level_dataset.csv"
CHUNK_OUT = ROOT / "Dataset" / "chunk_level_dataset.csv"

CHUNK_SIZE = 1000


def _as_int(x: object) -> int:
    return int(cast(Any, x))


def main() -> None:
    df = pd.read_csv(BOOK_LIST)

    book_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        book_id = _as_int(row["gutenberg_id"])
        path = CLEAN_DIR / f"{book_id}.txt"

        if not path.exists():
            continue

        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()

        words = text.split()

        # book-level dataset
        book_rows.append({
            "book_id": book_id,
            "author": row["author"],
            "era": row["era"],
            "text": text
        })

        # chunk-level dataset
        idx = 0
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = words[i:i + CHUNK_SIZE]

            if len(chunk) < 800:
                continue

            chunk_rows.append({
                "book_id": book_id,
                "author": row["author"],
                "era": row["era"],
                "chunk_id": f"{book_id}_{idx}",
                "text": " ".join(chunk)
            })

            idx += 1

    pd.DataFrame(book_rows).to_csv(BOOK_OUT, index=False)
    pd.DataFrame(chunk_rows).to_csv(CHUNK_OUT, index=False)

    print("done")


if __name__ == "__main__":
    main()
