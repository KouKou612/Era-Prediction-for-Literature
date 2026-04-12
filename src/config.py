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
RANDOM_STATE = 612

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
    "max_words": 10000,
}