import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "Dataset" / "selected_books.csv"
OUT_DIR = ROOT / "Dataset" / "books_raw"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _as_int(x: object) -> int:
    return int(cast(Any, x))


def get_text(book_id):
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except:
            pass

    return None


df = pd.read_csv(INPUT)

for _, row in df.iterrows():
    book_id = _as_int(row["gutenberg_id"])
    path = OUT_DIR / f"{book_id}.txt"

    if path.exists():
        continue

    text = get_text(book_id)

    if text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("ok", book_id)
    else:
        print("fail", book_id)

    time.sleep(1)