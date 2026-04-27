from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_suite
from config import RANDOM_STATE, TFIDF_CONFIG


def main():
    models = [
        (
            "TF-IDF + Majority (baseline)",
            Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
                ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
            ]),
        ),
        (
            "TF-IDF + LogisticRegression",
            Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
        ),
        (
            "TF-IDF + LinearSVC",
            Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
        ),
    ]
    run_era_suite("train_tfidf_compare", models)


if __name__ == "__main__":
    main()
