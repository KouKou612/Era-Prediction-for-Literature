import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.util import ngrams as nltk_ngrams
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_decade_suite
from config import RANDOM_STATE, NGRAM_COUNT_CONFIG

NR = NGRAM_COUNT_CONFIG["ngram_range"]


def ngram_analyzer(doc):
    lo, hi = int(NR[0]), int(NR[1])
    tokens = word_tokenize(doc.lower())
    out = []
    for n in range(lo, hi + 1):
        for gram in nltk_ngrams(tokens, n):
            out.append(" ".join(gram))
    return out


def build_ngram_logreg():
    return Pipeline([
        (
            "ngrams",
            CountVectorizer(
                analyzer=ngram_analyzer,
                max_features=NGRAM_COUNT_CONFIG["max_features"],
                min_df=NGRAM_COUNT_CONFIG["min_df"],
                binary=NGRAM_COUNT_CONFIG["binary"],
            ),
        ),
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
        (
            "ngrams",
            CountVectorizer(
                analyzer=ngram_analyzer,
                max_features=NGRAM_COUNT_CONFIG["max_features"],
                min_df=NGRAM_COUNT_CONFIG["min_df"],
                binary=NGRAM_COUNT_CONFIG["binary"],
            ),
        ),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main():
    models = [
        ("Word n-gram counts + LogisticRegression", build_ngram_logreg()),
        ("Word n-gram counts + LinearSVC", build_ngram_svm()),
    ]
    run_era_decade_suite("train_ngram_compare", models)


if __name__ == "__main__":
    main()
