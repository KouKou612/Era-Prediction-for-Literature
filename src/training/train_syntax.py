"""POS / syntactic feature channel only (spaCy coarse POS + scalars)."""

import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from train_common import run_era_decade_suite
from syntactic_features import SyntacticFeatureExtractor, spacy_model_available
from config import RANDOM_STATE, SYNTACTIC_CONFIG


def _syntax_extractor() -> SyntacticFeatureExtractor:
    return SyntacticFeatureExtractor(
        model_name=SYNTACTIC_CONFIG["spacy_model"],
        batch_size=SYNTACTIC_CONFIG["batch_size"],
        max_chars=SYNTACTIC_CONFIG["max_chars"],
    )


def build_syntax_logreg():
    return Pipeline([
        ("syntax", _syntax_extractor()),
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_syntax_svm():
    return Pipeline([
        ("syntax", _syntax_extractor()),
        ("scale", StandardScaler()),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main() -> None:
    model_name_cfg = SYNTACTIC_CONFIG["spacy_model"]
    if not spacy_model_available(model_name_cfg):
        print(
            f"spaCy model {model_name_cfg!r} not found. "
            "Install: pip install spacy && python -m spacy download "
            f"{model_name_cfg}",
            file=sys.stderr,
        )
        sys.exit(1)

    models: list[tuple[str, object]] = [
        ("POS / syntactic + LogisticRegression", build_syntax_logreg()),
        ("POS / syntactic + LinearSVC", build_syntax_svm()),
    ]
    run_era_decade_suite("train_syntax_compare", models)


if __name__ == "__main__":
    main()
