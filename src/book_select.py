from pathlib import Path
from typing import cast

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "Dataset" / "gutenberg_publication_years.csv"
OUTPUT = ROOT / "Dataset" / "selected_books.csv"

N_PER_ERA = 30


def assign_era(year):
    if 1700 <= year < 1798:
        return "Age of Reason"
    elif 1798 <= year < 1837:
        return "Romantic"
    elif 1837 <= year < 1901:
        return "Victorian"
    elif 1901 <= year < 1939:
        return "Modernist"
    return None


df = pd.read_csv(INPUT)
df.columns = [c.lower() for c in df.columns]

# basic cleaning
df = df.dropna(subset=["gutenberg_id", "author", "title", "publication_year"])
df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df = df.dropna(subset=["publication_year"])
df["publication_year"] = df["publication_year"].astype(int)

df["gutenberg_id"] = pd.to_numeric(df["gutenberg_id"], errors="coerce")
df = df.dropna(subset=["gutenberg_id"])
df["gutenberg_id"] = df["gutenberg_id"].astype(int)

# assign era
df["era"] = df["publication_year"].apply(assign_era)
df = cast(pd.DataFrame, df[df["era"].notnull()])

# keep one book per author per era
df = df.drop_duplicates(subset=["era", "author"])

# sample books
df_sample = cast(
    pd.DataFrame,
    df.groupby("era", group_keys=False).sample(n=N_PER_ERA, random_state=42),
)

df_sample.to_csv(OUTPUT, index=False)

print(df_sample["era"].value_counts())