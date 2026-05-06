import sys

import spacy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_suite
from syntactic_features import SyntacticFeatureExtractor
from config import RANDOM_STATE, SYNTACTIC_CONFIG


def main():
    model_name_cfg = SYNTACTIC_CONFIG["spacy_model"]
    try:
        spacy.load(model_name_cfg, disable=["ner", "lemmatizer", "attribute_ruler"])
    except OSError:
        print(
            f"spaCy model {model_name_cfg!r} not found. "
            "Install: pip install spacy && python -m spacy download "
            f"{model_name_cfg}",
            file=sys.stderr,
        )
        sys.exit(1)

    models = [
        (
            "POS / syntactic + LogisticRegression",
            Pipeline([
                (
                    "syntax",
                    SyntacticFeatureExtractor(
                        model_name=SYNTACTIC_CONFIG["spacy_model"],
                        batch_size=SYNTACTIC_CONFIG["batch_size"],
                    ),
                ),
                ("scale", StandardScaler()),
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
            "POS / syntactic + LinearSVC",
            Pipeline([
                (
                    "syntax",
                    SyntacticFeatureExtractor(
                        model_name=SYNTACTIC_CONFIG["spacy_model"],
                        batch_size=SYNTACTIC_CONFIG["batch_size"],
                    ),
                ),
                ("scale", StandardScaler()),
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
    run_era_suite("train_syntax_compare", models)


if __name__ == "__main__":
    main()
