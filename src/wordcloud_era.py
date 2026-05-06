from pathlib import Path
import argparse
import os
import ssl

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import nltk

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib import colormaps
from wordcloud import STOPWORDS, WordCloud

from config import ERA_CSV, ERA_TEXT_DIR, TRAIN_CONFIG, RANDOM_STATE
from data_utils import load_dataset


ERA_ORDER = ["Age of Reason", "Romantic", "Victorian", "Modernist", "Postmodern"]
ERA_COLORMAPS = {
    "Age of Reason": "Blues",
    "Romantic": "Purples",
    "Victorian": "Greens",
    "Modernist": "Oranges",
    "Postmodern": "Reds",
}

punct = (
    ". , : ! ? ; - * _ -- --- "
    "\u201c \u201d \u2019 \u2010 \u2011 \u2012 \u2013 \u2014 \u2015 "
    "\u2212 \u2500 \u2501 \u203e \u00af \u00ad"
).split()

stoplist = set(stopwords.words("english"))
stoplist.update(punct)
stoplist.update(
    "I It The And He upon one would could".split(),
)

# extra high-frequency Gutenberg junk
stoplist.update(
    {
        "said",
        "will",
        "may",
        "much",
        "little",
        "two",
        "three",
        "now",
        "made",
        "make",
        "come",
        "came",
        "well",
        "every",
        "many",
        "first",
        "time",
        "day",
        "good",
        "must",
        "never",
        "still",
        "back",
        "found",
        "see",
        "say",
        "know",
        "even",
        "long",
        "man",
        "men",
        "way",
        "hand",
        "eye",
        "old",
        "new",
        "great",
        "part",
        "place",
        "thing",
        "thought",
        "without",
        "take",
        "word",
        "life",
        "year",
        "right",
        "yet",
        "u",
        "i",
        "us",
    }
)

ALL_STOPWORDS = stoplist | set(STOPWORDS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/era_wordclouds")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_words = args.max_words

    for era in ERA_ORDER:
        era_texts = df.loc[df["era"] == era, "text"].tolist()
        text = " ".join(era_texts).strip()
        out = output_dir / (era.lower().replace(" ", "_") + "_wordcloud.png")
        cmap = ERA_COLORMAPS.get(era, "viridis")

        if not text:
            print(f"skip {era} (no text)")
            continue

        wc = WordCloud(
            width=900,
            height=650,
            background_color="white",
            max_words=max_words,
            stopwords=ALL_STOPWORDS,
            colormap=cmap,
        ).generate(text)

        c = colormaps.get_cmap(cmap)(0.98)
        title_color = (float(c[0]), float(c[1]), float(c[2]))

        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.title(era, fontweight="bold", fontsize=16, color=title_color)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print(out)


if __name__ == "__main__":
    main()
