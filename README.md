# Era Prediction for Literature

CSCI3349.01 NLP Final Project

Predict the **literary era** of an English Literature Work by eras(Age of Reason, Romantic, Victorian, Modernist, Postmodern) from a text chunk sampled from Project Gutenberg.

## Task

Given a ~10,000-character excerpt from a book, classify it into one of five eras:

| Era | Years |
|---|---|
| Age of Reason | 1700–1798 |
| Romantic | 1798–1837 |
| Victorian | 1837–1901 |
| Modernist | 1901–1945 |
| Postmodern | 1945–1974 |

## Models

| Model | Type | Script |
|---|---|---|
| TF-IDF + Majority  | Classic ML (Baseline) | `train_tfidf.py` |
| TF-IDF + LogReg / LinearSVC | Classic ML | `train_tfidf.py` |
| Word n-gram + LogReg / LinearSVC | Classic ML | `train_ngram.py` |
| POS / Syntactic + LogReg / LinearSVC | Classic ML | `train_syntax.py` |
| TextCNN | Neural | `train_CNN.py` |
| DistilBERT (fine-tuned) | Transformer | `train_BERT.py` |
| DeepSeek V4 Pro | LLM zero-shot | `deepseek_V4_pro.py` |
| Kimi K2.6 | LLM zero-shot | `kimi.py` |

## Project Layout

```
Era-Prediction-for-Literature/
├── Dataset/
│   ├── gutenberg_publication_years.csv   # raw metadata
│   ├── sample_by_era.csv                 # sampled book list (500 books)
│   ├── sample_by_era_with_chunks.csv     # pre-extracted chunks
│   ├── era_sample_raw/                   # downloaded Gutenberg texts
│   └── era_sample_clean/                 # cleaned texts
├── src/
│   ├── config.py                         # all hyperparameters
│   ├── data_utils.py                     # load_dataset (text loading + chunking)
│   ├── evaluation.py                     # metrics + classification report
│   ├── logging_utils.py                  # standard output logger (Tee)
│   ├── wordcloud_era.py                  # word cloud visualization
│   ├── preprocessing/
│   │   ├── book_select.py                # sample books from metadata
│   │   ├── book_download.py              # download from Project Gutenberg
│   │   ├── book_clean.py                 # strip headers/footers, book filter
│   │   └── make_chunked_csv.py           # extract chunks
│   └── training/
│       ├── train_common.py               # shared training loop
│       ├── syntactic_features.py         # spaCy POS/syntactic feature extractor
│       ├── train_tfidf.py
│       ├── train_ngram.py
│       ├── train_syntax.py
│       ├── train_CNN.py
│       ├── train_BERT.py
│       ├── deepseek_V4_pro.py
│       └── kimi.py
└── requirements.txt
```

## Setup

Create and activate a virtual environment from the **project root**:

```bash
cd Era-Prediction-for-Literature
python3 -m venv venv
source venv/bin/activate        
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline

### 1. Build the dataset

```bash
python3 src/preprocessing/book_select.py    # sample metadata -> sample_by_era.csv
python3 src/preprocessing/book_download.py  # download texts -> era_sample_raw/
python3 src/preprocessing/book_clean.py     # clean texts -> era_sample_clean/
python3 src/preprocessing/make_chunked_csv.py  # extract chunks -> sample_by_era_with_chunks.csv
```

### 2. Train classic ML and neural models

Run from the **project root**:

```bash
python3 src/training/train_tfidf.py
python3 src/training/train_ngram.py
python3 src/training/train_syntax.py
python3 src/training/train_CNN.py
python3 src/training/train_BERT.py
```

Logs are saved as name & timestamped `.txt` files in `src/training/`.

### 3. Run LLM zero-shot inference

Set your API key, then run from the **project root**:

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your_key"
python3 src/training/deepseek_V4_pro.py

# Kimi
export KIMI_API_KEY="your_key"
python3 src/training/kimi.py
```

Results are saved to `Dataset/deepseek_results.csv` and `Dataset/kimi_results.csv`.

## Key Configuration (`src/config.py`)

| Parameter | Value | Effect |
|---|---|---|
| `RANDOM_STATE` | 612 | Train/test split seed (fixed state for comparison)|
| `random_chunk_chars` | 10,000 | Characters per book chunk |
| `chunk_random_state` | 42 | Chunk sampling seed |
| `test_size` | 0.2 | 80/20 train/test split |

## Text Sampling

- **Classic ML + BERT**: a random contiguous 10,000-character chunk is drawn from each book at load time (seed=42), roughly 1,500–2,000 words (~3–4 pages).
- **CNN**: takes the first 10,000 words of each book (no random sampling).
- **LLMs**: use pre-extracted chunks from `sample_by_era_with_chunks.csv`; no train/test split (zero-shot over all 500 books).
