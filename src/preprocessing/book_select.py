from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = ROOT / "Dataset" / "gutenberg_publication_years.csv"

N_PER_ERA = 100
RANDOM_STATE = 42


def assign_era(year):
    if 1700 <= year < 1798:
        return "Age of Reason"
    elif 1798 <= year < 1837:
        return "Romantic"
    elif 1837 <= year < 1901:
        return "Victorian"
    elif 1901 <= year < 1945:
        return "Modernist"
    elif 1945 <= year <= 1974:
        return "Postmodern"
    return None


df = pd.read_csv(INPUT)
df.columns = [c.lower() for c in df.columns]
df = df.drop(columns=["lccn", "wikipedia_url"])

df = df.dropna(subset=["gutenberg_id", "author", "title", "publication_year"]).copy()

df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df = df.dropna(subset=["publication_year"]).copy()
df["publication_year"] = df["publication_year"].astype(int)

df["gutenberg_id"] = pd.to_numeric(df["gutenberg_id"], errors="coerce")
df = df.dropna(subset=["gutenberg_id"]).copy()
df["gutenberg_id"] = df["gutenberg_id"].astype(int)

df = df[(df["publication_year"] >= 1700) & (df["publication_year"] <= 1974)].copy()

df["era"] = df["publication_year"].apply(assign_era)

df = df[df["era"].notna()].copy()

df = df.sample(frac=1, random_state=RANDOM_STATE).copy()

df_era = df.drop_duplicates(subset=["era", "author"]).copy()
df_era_sample = pd.concat(
    [
        g.sample(n=min(len(g), N_PER_ERA), random_state=RANDOM_STATE)
        for _, g in df_era.groupby("era")
    ],
    ignore_index=True,
)

df_era_sample.to_csv(ROOT / "Dataset" / "sample_by_era.csv", index=False)

print("Books per era:")
print(df_era_sample["era"].value_counts().sort_index())
