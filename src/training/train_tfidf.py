"""TF-IDF feature channel only (lexical baseline + linear classifiers)."""

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_decade_suite
from config import RANDOM_STATE, TFIDF_CONFIG


def build_tfidf_majority():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
    ])


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


def main() -> None:
    models: list[tuple[str, object]] = [
        ("TF-IDF + Majority (baseline)", build_tfidf_majority()),
        ("TF-IDF + LogisticRegression", build_tfidf_logreg()),
        ("TF-IDF + LinearSVC", build_tfidf_svm()),
    ]
    run_era_decade_suite("train_tfidf_compare", models)


if __name__ == "__main__":
    main()
