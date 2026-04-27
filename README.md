# Era-Prediction-for-Literature

CSCI3349.01 Final Project

Ray's Update Version 1.0.3

Predict **literary era** (Age of Reason, Romantic, Victorian, Modernist, Postmodern) from English prose sampled from Project Gutenberg. The project no longer trains or evaluates a separate **decade** label; preprocessing and training scripts are **era-only**.

## Layout

- **`src/preprocessing/`** — build the corpus: sampling CSV, Gutenberg download, header/footer cleaning and quality filter (`book_select.py`, `book_download.py`, `book_clean.py`).
- **`src/training/`** — train and evaluate models on the **era** task: `train_tfidf.py`, `train_syntax.py`, `train_ngram.py`, shared loop in `train_common.py` (`run_era_suite`), spaCy features in `syntactic_features.py`. Optional Hugging Face: `train_BERT.py` (DistilBERT). When logging is enabled, run logs are timestamped `*.txt` under `src/training/`.
- **`src/`** (top level) — shared settings and helpers: `config.py`, `data_utils.py`, `evaluation.py`, `logging_utils.py`.

From the repo root, use `python3` with paths like `src/preprocessing/book_select.py` or `src/training/train_tfidf.py`.

## Pipeline (era only)

### 1) Install requirements

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install nltk spacy
python3 -m spacy download en_core_web_sm
```

### 2) Build dataset

```bash
# sample metadata -> Dataset/sample_by_era.csv
python3 src/preprocessing/book_select.py

# download Gutenberg texts -> Dataset/era_sample_raw/
python3 src/preprocessing/book_download.py

# clean texts -> Dataset/era_sample_clean/
python3 src/preprocessing/book_clean.py
```

### 3) Train classic ML models

```bash
# TF-IDF channel
python3 src/training/train_tfidf.py

# Word n-gram channel (NLTK-based analyzer)
python3 src/training/train_ngram.py

# Syntactic/POS channel (spaCy)
python3 src/training/train_syntax.py
```

### 4) (Optional) DistilBERT fine-tuning

```bash
python3 src/training/train_BERT.py
```

### 5) Run all common steps in one shot

```bash
python3 -m pip install -r requirements.txt && \
python3 -m pip install nltk spacy && \
python3 -m spacy download en_core_web_sm && \
python3 src/preprocessing/book_select.py && \
python3 src/preprocessing/book_download.py && \
python3 src/preprocessing/book_clean.py && \
python3 src/training/train_tfidf.py && \
python3 src/training/train_ngram.py && \
python3 src/training/train_syntax.py
```

Shared behavior: **`TRAIN_CONFIG`** uses a fixed-seed **random 10,000-character slice per book** in `data_utils.load_dataset` (unless you change `random_chunk_chars`), so TF-IDF, n-grams, and syntax see comparable text budgets.

## Changelog (high level)

- **1.0.3** — Dropped `train_transformer.py` and `TRANSFORMER_CONFIG`; Hugging Face fine-tuning is **`train_BERT.py` only** (DistilBERT).
- **1.0.2** — Removed the **decade** task end-to-end: no `sample_by_decade.csv`, no `decade_sample_*` paths in config or scripts, no decade metrics. Evaluation reports **era** labels in timeline order. Decade-specific corpus folders were dropped from version control; the repo is **era prediction only**.
- **1.0.1** — Split `src/preprocessing/` vs `src/training/`, unified random 10k-char loading, syntax/n-gram improvements (NLTK n-grams in custom analyzer, etc.). See git history for detail.
