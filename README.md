# Era-Prediction-for-Literature

CSCI3349.01 Final Project

## Environment

From the repository root (`Era-Prediction-for-Literature/`), activate your virtual environment, then:

```bash
pip install -r requirements.txt
```

Use a recent Python (3.10+; 3.14 was used in development).

Paths inside the pipeline scripts are resolved from each file’s location, so you can run them as `python src/<script>.py` from the repo root regardless of your shell’s current directory naming (e.g. `src/` vs root).

## Pipeline (run in order)

1. **Select books** — metadata → balanced sample per era (one book per author per era)  
   `python src/book_select.py`  
   Output: `Dataset/selected_books.csv`

2. **Download full texts** — Gutenberg, IDs from the selection only  
   `python src/book_download.py`  
   Output: `Dataset/books_raw/*.txt`  
   (Polite rate limiting: one request per second between books.)

3. **Strip Project Gutenberg boilerplate** — header/footer markers  
   `python src/book_clean.py`  
   Output: `Dataset/books_clean/*.txt`

4. **Build datasets** — whole-book rows and ~1000-word chunks  
   `python src/make_datasets.py`  
   Output: `Dataset/book_level_dataset.csv`, `Dataset/chunk_level_dataset.csv`

5. **Train / test split** — by **author** globally (same author never appears in both splits, including across eras)  
   `python src/make_split.py`  
   Output: `Dataset/train_chunks.csv`, `Dataset/test_chunks.csv`, and when book-level data exists: `Dataset/train_books.csv`, `Dataset/test_books.csv`

To rerun from scratch, delete the generated `books_raw`, `books_clean`, and downstream CSVs you want to rebuild; keep `Dataset/gutenberg_publication_years.csv` unless you replace the metadata source.

## What we changed today

- **Stable paths**: `book_clean.py`, `make_datasets.py`, and `make_split.py` now anchor `Dataset/` to the project root via `Path(__file__)`, consistent with `book_select.py` and `book_download.py`, so runs from `src/` or the repo root behave the same.
- **Train/test split**: Replaced per-era author splits (which could put the same author name in both train and test when they had books in multiple eras) with a **single global author-level** `train_test_split`, and added assertions for zero overlap on authors and `book_id`.
- **Book-level splits**: `make_split.py` also writes `train_books.csv` / `test_books.csv` using the same author assignment, so TF-IDF (book-level) and BERT-style (chunk-level) experiments share one leakage-safe partition.
- **Dependencies / tooling**: Documented `requirements.txt` usage; local venv installs for `pandas`, `requests`, and `scikit-learn` as needed for the pipeline and type checking.
