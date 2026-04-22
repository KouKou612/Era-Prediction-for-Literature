"""Word n-gram count feature channel only (CountVectorizer; no TF-IDF)."""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_decade_suite
from config import RANDOM_STATE, NGRAM_COUNT_CONFIG


def build_ngram_logreg():
    return Pipeline([
        ("ngrams", CountVectorizer(**NGRAM_COUNT_CONFIG)),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_ngram_svm():
    return Pipeline([
        ("ngrams", CountVectorizer(**NGRAM_COUNT_CONFIG)),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main() -> None:
    models: list[tuple[str, object]] = [
        ("Word n-gram counts + LogisticRegression", build_ngram_logreg()),
        ("Word n-gram counts + LinearSVC", build_ngram_svm()),
    ]
    run_era_decade_suite("train_ngram_compare", models)


if __name__ == "__main__":
    main()
