from pathlib import Path
from typing import cast

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "Dataset" / "gutenberg_publication_years.csv"
OUTPUT = ROOT / "Dataset" / "selected_books.csv"


N_PER_ERA = 50
N_PER_DECADE = 20
RANDOM_STATE = 612

def assign_era(year):
    if 1700 <= year < 1798:
        return "Age of Reason"
    elif 1798 <= year < 1837:
        return "Romantic"
    elif 1837 <= year < 1901:
        return "Victorian"
    elif 1901 <= year < 1945:
        return "Modernist"
    elif 1945 <= year <= 1974: # 1974 is the latest book's publication year in the dataset
        return "Postmodern"
    return None

def assign_decade(year):
    decade_start = (year // 10) * 10
    return f"{decade_start}s"

df = pd.read_csv(INPUT)
df.columns = [c.lower() for c in df.columns]
df = df.drop(columns=["lccn", "wikipedia_url"])

# check for missing values and convert types
df = df.dropna(subset=["gutenberg_id", "author", "title", "publication_year"]).copy()

# convert publication_year to numeric, coercing errors to NaN, then drop rows with NaN
df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df = df.dropna(subset=["publication_year"]).copy()
df["publication_year"] = df["publication_year"].astype(int)

# convert gutenberg_id to numeric, coercing errors to NaN, then drop rows with NaN
df["gutenberg_id"] = pd.to_numeric(df["gutenberg_id"], errors="coerce")
df = df.dropna(subset=["gutenberg_id"]).copy()
df["gutenberg_id"] = df["gutenberg_id"].astype(int)

# remove books published before 1700 and after 1974
df = df[(df["publication_year"] >= 1700) & (df["publication_year"] <= 1974)].copy()

# assign era
df["era"] = df["publication_year"].apply(assign_era)
# assign decade
df["decade"] = df["publication_year"].apply(assign_decade)

# remove rows with invalid era just in case
df = df[df["era"].notna()].copy()

# shuffle once to avoid order bias before dedup
df = df.sample(frac=1, random_state=RANDOM_STATE).copy()

# build era dataset
df_era = df.drop_duplicates(subset=["era", "author"]).copy()
df_era_sample = pd.concat(
    [
        g.sample(n=min(len(g), N_PER_ERA), random_state=RANDOM_STATE)
        for _, g in df_era.groupby("era")
    ],
    ignore_index=True,
)

# build decade dataset
df_decade = df.drop_duplicates(subset=["decade", "author"]).copy()
df_decade_sample = pd.concat(
    [
        g.sample(n=min(len(g), N_PER_DECADE), random_state=RANDOM_STATE)
        for _, g in df_decade.groupby("decade")
    ],
    ignore_index=True,
)


# save
df_era_sample.to_csv(ROOT / "Dataset" / "sample_by_era.csv", index=False)
df_decade_sample.to_csv(ROOT / "Dataset" / "sample_by_decade.csv", index=False)


# print counts
print("Books per era:")
print(df_era_sample["era"].value_counts().sort_index())

print("\nBooks per decade:")
print(df_decade_sample["decade"].value_counts().sort_index())