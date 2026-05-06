# export DEEPSEEK_API_KEY="your_api_key_here"

import os
import sys
from pathlib import Path
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from train_common import TRAINING_DIR

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import evaluate_model
from logging_utils import start_logging

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("please set the DEEPSEEK_API_KEY environment variable")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

INPUT_CSV = Path("Dataset/sample_by_era_with_chunks.csv")
OUTPUT_CSV = Path("Dataset/deepseek_results.csv")

LABELS = [
    "Age of Reason",
    "Romantic",
    "Victorian",
    "Modernist",
    "Postmodern",
]


def process_one(i, row):
    chunk = row["chunk"]
    true_era = row["era"]
    prompt = f"""
            Classify the literary era of the following text.

            Choose one:
            Age of Reason, Romantic, Victorian, Modernist, Postmodern

            Note:
            - The text is a random contiguous chunk from a book, not necessarily the beginning.
            - In this dataset, Age of Reason = 1700-1798, Romantic = 1798-1837, Victorian = 1837-1901, Modernist = 1901-1945, Postmodern = 1945-present.

            Text:
            {chunk}

            Answer with ONLY the era name.
            """
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You are a literary scholar specializing in classifying the era of English literature."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        raw = response.choices[0].message.content
        pred = None
        if raw:
            for lab in LABELS:
                if lab.lower() in raw.lower():
                    pred = lab
                    break
        return i, pred, true_era
    except Exception as e:
        print(f"Error occurred while processing row {i+1}: {e}")
        return i, None, true_era


def main():
    start_logging("deepseek_run", log_dir=TRAINING_DIR)

    df = pd.read_csv(INPUT_CSV)

    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_one, i, row): i for i, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            idx, pred, true_era = future.result()
            results[idx] = pred
            print(f"True: {true_era} | Pred: {pred}")

    df["prediction"] = results
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved to:", OUTPUT_CSV)

    df_eval = df.dropna(subset=["prediction"])

    y_true = df_eval["era"].tolist()
    y_pred = df_eval["prediction"].tolist()

    print("\n=== EVALUATION ===")
    evaluate_model(y_true, y_pred, "era")


if __name__ == "__main__":
    main()
