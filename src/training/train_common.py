import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sklearn.model_selection import train_test_split

from data_utils import load_dataset
from evaluation import evaluate_model, get_metrics
from logging_utils import start_logging
from config import (
    ERA_CSV,
    DECADE_CSV,
    ERA_TEXT_DIR,
    DECADE_TEXT_DIR,
    RANDOM_STATE,
    TRAIN_CONFIG,
)


def train_and_evaluate(df, label_col, model, model_name):
    X = df["text"]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_CONFIG["test_size"],
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} | {label_col.upper()} ===")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    evaluate_model(y_test, y_pred, label_col)

    metrics = get_metrics(y_test, y_pred, label_col)
    print("\nMetrics dictionary:")
    print(metrics)

    return metrics


def run_models_for_task(
    df,
    label_col: str,
    models: list[tuple[str, object]],
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for model_name, model in models:
        metrics = train_and_evaluate(df, label_col, model, model_name)
        results[model_name] = metrics
        print("\n" + "-" * 60 + "\n")
    return results


def run_era_decade_suite(
    log_stem: str,
    models: list[tuple[str, object]],
    *,
    text_prefix_chars: int | None = None,
) -> None:
    start_logging(log_stem, log_dir=TRAINING_DIR)

    random_chunk = TRAIN_CONFIG.get("random_chunk_chars")
    chunk_seed = TRAIN_CONFIG.get("chunk_random_state", RANDOM_STATE)

    if text_prefix_chars is not None and random_chunk is not None:
        raise ValueError("Pass text_prefix_chars only when TRAIN_CONFIG['random_chunk_chars'] is None.")

    if random_chunk is not None:
        max_words = None
        prefix_chars = None
        print(
            f"Loading ERA dataset (random contiguous {random_chunk} chars per book, seed={chunk_seed})...",
        )
        era_df = load_dataset(
            ERA_CSV,
            ERA_TEXT_DIR,
            "era",
            max_words=max_words,
            prefix_chars=prefix_chars,
            random_chunk_chars=random_chunk,
            random_state=chunk_seed,
        )
    elif text_prefix_chars is not None:
        max_words = None
        print(
            f"Loading ERA dataset (first {text_prefix_chars} characters per book)...",
        )
        era_df = load_dataset(
            ERA_CSV,
            ERA_TEXT_DIR,
            "era",
            max_words=max_words,
            prefix_chars=text_prefix_chars,
        )
    else:
        max_words = TRAIN_CONFIG["max_words"]
        print("Loading ERA dataset...")
        era_df = load_dataset(
            ERA_CSV,
            ERA_TEXT_DIR,
            "era",
            max_words=max_words,
        )

    print("\nRunning ERA models...\n")
    era_results = {}
    for model_name, model in models:
        era_results[model_name] = train_and_evaluate(era_df, "era", model, model_name)
        print("\n" + "-" * 60 + "\n")

    print("\n" + "=" * 80 + "\n")

    if random_chunk is not None:
        print(
            f"Loading DECADE dataset (random contiguous {random_chunk} chars per book, seed={chunk_seed})...",
        )
        decade_df = load_dataset(
            DECADE_CSV,
            DECADE_TEXT_DIR,
            "decade",
            max_words=None,
            prefix_chars=None,
            random_chunk_chars=random_chunk,
            random_state=chunk_seed,
        )
    elif text_prefix_chars is not None:
        print(
            f"Loading DECADE dataset (first {text_prefix_chars} characters per book)...",
        )
        decade_df = load_dataset(
            DECADE_CSV,
            DECADE_TEXT_DIR,
            "decade",
            max_words=None,
            prefix_chars=text_prefix_chars,
        )
    else:
        print("Loading DECADE dataset...")
        decade_df = load_dataset(
            DECADE_CSV,
            DECADE_TEXT_DIR,
            "decade",
            max_words=max_words,
        )

    print("\nRunning DECADE models...\n")
    decade_results = {}
    for model_name, model in models:
        decade_results[model_name] = train_and_evaluate(decade_df, "decade", model, model_name)
        print("\n" + "-" * 60 + "\n")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nERA RESULTS:")
    for model_name, metrics in era_results.items():
        print(f"{model_name}: {metrics}")

    print("\nDECADE RESULTS:")
    for model_name, metrics in decade_results.items():
        print(f"{model_name}: {metrics}")
