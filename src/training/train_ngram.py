import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.util import ngrams as nltk_ngrams
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_suite
from config import RANDOM_STATE, NGRAM_COUNT_CONFIG

NR = NGRAM_COUNT_CONFIG["ngram_range"]


def main():
    lo, hi = int(NR[0]), int(NR[1])

    def ngram_analyzer(doc):
        tokens = word_tokenize(doc.lower())
        out = []
        for n in range(lo, hi + 1):
            for gram in nltk_ngrams(tokens, n):
                out.append(" ".join(gram))
        return out

    vec_kw = dict(
        analyzer=ngram_analyzer,
        max_features=NGRAM_COUNT_CONFIG["max_features"],
        min_df=NGRAM_COUNT_CONFIG["min_df"],
        binary=NGRAM_COUNT_CONFIG["binary"],
    )

    models = [
        (
            "Word n-gram counts + LogisticRegression",
            Pipeline([
                ("ngrams", CountVectorizer(**vec_kw)),
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
            "Word n-gram counts + LinearSVC",
            Pipeline([
                ("ngrams", CountVectorizer(**vec_kw)),
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
    run_era_suite("train_ngram_compare", models)


if __name__ == "__main__":
    main()
