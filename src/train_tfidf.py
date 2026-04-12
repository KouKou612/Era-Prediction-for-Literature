from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from data_utils import load_dataset
from evaluation import evaluate_model, get_metrics
from logging_utils import start_logging
from config import (
    ERA_CSV,
    DECADE_CSV,
    ERA_TEXT_DIR,
    DECADE_TEXT_DIR,
    RANDOM_STATE,
    TFIDF_CONFIG,
    TRAIN_CONFIG,
)


def build_tfidf_logreg():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_tfidf_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def train_and_evaluate(df, label_col: str, model, model_name: str) -> dict:
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


def run_models_for_task(df, label_col: str) -> dict:
    results = {}

    models = [
        ("TF-IDF + LogisticRegression", build_tfidf_logreg()),
        ("TF-IDF + LinearSVC", build_tfidf_svm()),
    ]

    for model_name, model in models:
        metrics = train_and_evaluate(df, label_col, model, model_name)
        results[model_name] = metrics
        print("\n" + "-" * 60 + "\n")

    return results


def main() -> None:
    start_logging("train_tfidf_compare")

    print("Loading ERA dataset...")
    era_df = load_dataset(
        ERA_CSV,
        ERA_TEXT_DIR,
        "era",
        max_words=TRAIN_CONFIG["max_words"],
    )

    print("\nRunning ERA models...\n")
    era_results = run_models_for_task(era_df, "era")

    print("\n" + "=" * 80 + "\n")

    print("Loading DECADE dataset...")
    decade_df = load_dataset(
        DECADE_CSV,
        DECADE_TEXT_DIR,
        "decade",
        max_words=TRAIN_CONFIG["max_words"],
    )

    print("\nRunning DECADE models...\n")
    decade_results = run_models_for_task(decade_df, "decade")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nERA RESULTS:")
    for model_name, metrics in era_results.items():
        print(f"{model_name}: {metrics}")

    print("\nDECADE RESULTS:")
    for model_name, metrics in decade_results.items():
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()