# Era-Prediction-for-Literature

CSCI3349.01 Final Project

Ray's Update Version 1.0.1

## Layout

- **`src/preprocessing/`** — build the corpus: sampling CSVs, Gutenberg download, header/footer cleaning and quality filter (`book_select.py`, `book_download.py`, `book_clean.py`).
- **`src/training/`** — train and evaluate models: one entry script per feature channel (`train_tfidf.py`, `train_syntax.py`, `train_ngram.py`), shared loop in `train_common.py`, spaCy transformer in `syntactic_features.py`. Run logs are written under `src/training/`.
- **`src/`** (top level) — shared settings and helpers: `config.py`, `data_utils.py`, `evaluation.py`, `logging_utils.py`.

From the repo root, use `python3` with paths like `src/preprocessing/book_select.py` or `src/training/train_tfidf.py`.

## WHAT I CHANGED

- `src/preprocessing/book_select.py`: N_PER_ERA is 100 instead of 50 (still one author per era, up to 100 per era). Run `python3 src/preprocessing/book_select.py` to refresh `Dataset/sample_by_era.csv` and `sample_by_decade.csv`. I didn't change anything about the decade yet

- `src/config.py` and `src/preprocessing/book_select.py`: RANDOM_STATE is 42 instead of 612 for sampling and training.

- `src/preprocessing/book_clean.py`: after normal Gutenberg header/footer removal, books that are too short (under 300 words) or have too high a fraction of lines that look like MIDI paths, copyright lines, or Producer lines are not written to `era_sample_clean` or `decade_sample_clean`; an old clean file with the same name is deleted if the book fails the check. Run `python3 src/preprocessing/book_clean.py` to apply this to your local clean folders.

- Training entry scripts: `python3 src/training/train_tfidf.py` (TF-IDF only), `python3 src/training/train_syntax.py` (spaCy POS / syntactic only; needs spacy + en_core_web_sm), `python3 src/training/train_ngram.py` (word n-gram counts only via CountVectorizer / NGRAM_COUNT_CONFIG). Shared loop lives in `src/training/train_common.py` (import-only module, not a script you run by itself).

## Ray's Update Version 1.0.1

- Split the repo into **`src/preprocessing/`** (select / download / clean) vs **`src/training/`** (three `train_*.py` scripts + `train_common.py` + `syntactic_features.py`) so the pipeline is easier to follow.
- **Same text budget for every channel:** `TRAIN_CONFIG` uses a fixed-seed **random 10,000-character slice per book** in `data_utils.load_dataset`, so TF-IDF, n-grams, and syntax all read the same kind of input (full file read, then one contiguous chunk).
- **Syntax (`syntactic_features.py`):** spaCy **UPOS fractions** plus **avg token length** and **avg sentence length**; dropped the unused `max_chars` cap on the transformer (length is controlled at load time only).
- **N-grams (`train_ngram.py`):** **NLTK `word_tokenize`** + **`nltk.ngrams`** inside a custom **`CountVectorizer` analyzer** so `ngram_range=(1,3)` actually applies (sklearn ignores `ngram_range` when `analyzer` is callable unless you build n-grams yourself). **No stop-word list** for that channel; added **`nltk`** to `requirements.txt` and **`nltk.download`** at the top of `train_ngram.py` for punkt data.
- **General:** simplified a lot of the Python (fewer helpers / type hints / long docstrings) while keeping **`RANDOM_STATE` / `chunk_random_state`** behavior so runs stay reproducible.