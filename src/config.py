from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"

ERA_CSV = DATASET_DIR / "sample_by_era.csv"

ERA_TEXT_DIR = DATASET_DIR / "era_sample_clean"

MODEL_OUTPUT_DIR = ROOT / "model_outputs"
RANDOM_STATE = 612

TFIDF_CONFIG = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
}

# random_chunk_chars: random contiguous slice per book (same for tfidf / ngram / syntax). None -> use max_words only.
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_chunk_chars": 10_000,
    "chunk_random_state": 42,
    "max_words": 10_000,
}

SYNTACTIC_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "batch_size": 32,
}

# ngram_range used in train_ngram.ngram_analyzer; CountVectorizer only gets max_features / min_df / binary.
NGRAM_COUNT_CONFIG = {
    "ngram_range": (1, 3),
    "max_features": 20000,
    "min_df": 2,
    "binary": False,
}
