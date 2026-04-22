from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"

# CSV paths
ERA_CSV = DATASET_DIR / "sample_by_era.csv"
DECADE_CSV = DATASET_DIR / "sample_by_decade.csv"

# Text directories
ERA_TEXT_DIR = DATASET_DIR / "era_sample_clean"
DECADE_TEXT_DIR = DATASET_DIR / "decade_sample_clean"

# Random seed
RANDOM_STATE = 42

# TF-IDF parameters
TFIDF_CONFIG = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
}

# Training config
TRAIN_CONFIG = {
    "test_size": 0.2,
    # If set, every train_* script loads full book text then takes one random contiguous
    # substring of this many characters per book (same for TF-IDF / n-gram / syntax).
    # Set to None to use max_words prefix-instead (legacy).
    "random_chunk_chars": 10_000,
    "chunk_random_state": 42,
    "max_words": 10_000,
}

# spaCy syntactic features (see training/syntactic_features.py)
SYNTACTIC_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "batch_size": 32,
    "max_chars": 120_000,
}

# Word n-gram counts (separate channel from TF-IDF; no idf weighting).
# Tune here when you swap in a different n-gram strategy later.
NGRAM_COUNT_CONFIG = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 3),
    "max_features": 20000,
    "min_df": 2,
    "binary": False,
}