from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CHUNK_IN = ROOT / "Dataset" / "chunk_level_dataset.csv"
BOOK_IN = ROOT / "Dataset" / "book_level_dataset.csv"

TRAIN_CHUNKS = ROOT / "Dataset" / "train_chunks.csv"
TEST_CHUNKS = ROOT / "Dataset" / "test_chunks.csv"
TRAIN_BOOKS = ROOT / "Dataset" / "train_books.csv"
TEST_BOOKS = ROOT / "Dataset" / "test_books.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(CHUNK_IN)

    # One global split per author name so the same author never appears in both train and test
    # (avoids leakage when one author has books in multiple eras).
    authors = sorted(df["author"].unique())
    train_auth, test_auth = train_test_split(
        authors,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    assert not set(train_auth) & set(test_auth)

    train_df = df[df["author"].isin(train_auth)].copy()
    test_df = df[df["author"].isin(test_auth)].copy()

    overlap_auth = set(train_df["author"]) & set(test_df["author"])
    assert len(overlap_auth) == 0

    tr_books = set(train_df["book_id"])
    te_books = set(test_df["book_id"])
    overlap_book = tr_books & te_books
    assert len(overlap_book) == 0

    train_df.to_csv(TRAIN_CHUNKS, index=False)
    test_df.to_csv(TEST_CHUNKS, index=False)

    print("chunk train/test rows:", len(train_df), len(test_df))
    print("train authors:", len(set(train_auth)), "test authors:", len(set(test_auth)))
    print("author overlap (should be 0):", len(overlap_auth))
    print("book_id overlap (should be 0):", len(overlap_book))

    if BOOK_IN.exists():
        books = pd.read_csv(BOOK_IN)
        train_books = books[books["author"].isin(train_auth)]
        test_books = books[books["author"].isin(test_auth)]
        train_books.to_csv(TRAIN_BOOKS, index=False)
        test_books.to_csv(TEST_BOOKS, index=False)
        print("book train/test rows:", len(train_books), len(test_books))
    else:
        print(f"skip book-level split (missing {BOOK_IN})")


if __name__ == "__main__":
    main()
