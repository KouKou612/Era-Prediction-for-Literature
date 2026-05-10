from pathlib import Path
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib import colormaps
from wordcloud import WordCloud

# share stoplist / era order with the count-based word cloud
import wordcloud_era as _wc

from config import ERA_CSV, ERA_TEXT_DIR, TRAIN_CONFIG, RANDOM_STATE
from data_utils import load_dataset

# final_project/outputs/tfidf WordCloud (parent of Era-Prediction-for-Literature repo)
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "outputs" / "tfidf WordCloud"
)

ERA_ORDER = _wc.ERA_ORDER
ERA_COLORMAPS = _wc.ERA_COLORMAPS
ALL_STOPWORDS = _wc.ALL_STOPWORDS


def era_tfidf_frequencies(vectorizer, X, df, era, max_terms):
    mask = df["era"].values == era
    if not mask.any():
        return {}
    sub = X[mask]
    weights = np.asarray(sub.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    pairs = []
    for term, w in zip(terms, weights):
        if w <= 0:
            continue
        key = term.lower() if isinstance(term, str) else str(term).lower()
        if key in ALL_STOPWORDS or len(key) < 2:
            continue
        pairs.append((term, float(w)))
    pairs.sort(key=lambda x: -x[1])
    return dict(pairs[:max_terms])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"default: {_DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument("--max-words", type=int, default=50)
    args = p.parse_args()

    chunk = TRAIN_CONFIG.get("random_chunk_chars")
    seed = TRAIN_CONFIG.get("chunk_random_state", RANDOM_STATE)

    if chunk is not None:
        df = load_dataset(
            ERA_CSV,
            ERA_TEXT_DIR,
            "era",
            max_words=None,
            random_chunk_chars=chunk,
            random_state=seed,
        )
    else:
        df = load_dataset(
            ERA_CSV,
            ERA_TEXT_DIR,
            "era",
            max_words=TRAIN_CONFIG.get("max_words", 10000),
        )

    texts = df["text"].tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 1),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_words = args.max_words
    pool = max(max_words * 4, 200)

    for era in ERA_ORDER:
        freq = era_tfidf_frequencies(vectorizer, X, df, era, pool)
        out = output_dir / (era.lower().replace(" ", "_") + "_wordcloud_tfidf.png")
        cmap = ERA_COLORMAPS.get(era, "viridis")

        if not freq:
            print(f"skip {era} (no terms)")
            continue

        wc = WordCloud(
            width=900,
            height=650,
            background_color="white",
            max_words=max_words,
            stopwords=ALL_STOPWORDS,
            colormap=cmap,
        ).generate_from_frequencies(freq)

        c = colormaps.get_cmap(cmap)(0.98)
        title_color = (float(c[0]), float(c[1]), float(c[2]))

        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.title(era + " (TF-IDF)", fontweight="bold", fontsize=16, color=title_color)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print(out)


if __name__ == "__main__":
    main()
