from pathlib import Path
import time
import requests
import pandas as pd
import config


"""
Download the full text of the selected books from Project Gutenberg using the gutenberg_id.
The books are in the era_sample.csv and decade_sample.csv files, which were created by book_select.py.
Skipping books that have already been downloaded to avoid redundant work.
"""



ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"

ERA_CSV = DATASET_DIR / "sample_by_era.csv"
DECADE_CSV = DATASET_DIR / "sample_by_decade.csv"

ERA_OUTDIR = DATASET_DIR / "era_sample_raw"
DECADE_OUTDIR = DATASET_DIR / "decade_sample_raw"

REQUEST_TIMEOUT = 20
SLEEP_SECONDS = 0.5


def gutenberg_text_urls(book_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-8.txt",
    ]


def download_text(book_id: int) -> str | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GutenbergBookDownloader/1.0)"
    }

    for url in gutenberg_text_urls(book_id):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
        except requests.RequestException:
            continue

    return None


def download_from_csv(csv_path: Path, outdir: Path) -> tuple[list[int], list[int], list[int]]:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"gutenberg_id", "title", "author"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    total = len(df)
    success = 0
    failed = 0
    skipped = 0

    downloaded_ids: list[int] = []
    skipped_ids: list[int] = []
    failed_ids: list[int] = []

    for i, row in df.iterrows():
        book_id = None
        try:
            book_id = int(row["gutenberg_id"])
            title = str(row["title"])
            author = str(row["author"])

            filename = f"{book_id}.txt"
            filepath = outdir / filename

            # Skip if the book file already exists
            if filepath.exists():
                print(f"[{i + 1}/{total}] Skip existing: {filename}")
                skipped += 1
                skipped_ids.append(book_id)
                continue

            text = download_text(book_id)

            if text is None:
                print(f"[{i + 1}/{total}] Failed: {book_id} | {title}")
                failed += 1
                failed_ids.append(book_id)
            else:
                filepath.write_text(text, encoding="utf-8")
                print(f"[{i + 1}/{total}] Downloaded: {filename}")
                success += 1
                downloaded_ids.append(book_id)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"[{i + 1}/{total}] Error on row {i}: {e}")
            failed += 1
            if book_id is not None:
                failed_ids.append(book_id)

    print(f"\nFinished {csv_path.name}")
    print(f"Saved to: {outdir}")
    print(f"Downloaded: {success}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {failed}")

    if failed_ids:
        print("\nFailed book IDs:")
        print(failed_ids)
    else:
        print("\nNo failed downloads.")

    return downloaded_ids, skipped_ids, failed_ids


def main() -> None:
    print("Downloading ERA dataset...\n")
    era_downloaded, era_skipped, era_failed = download_from_csv(ERA_CSV, ERA_OUTDIR)

    print("\n" + "=" * 60 + "\n")

    print("Downloading DECADE dataset...\n")
    decade_downloaded, decade_skipped, decade_failed = download_from_csv(DECADE_CSV, DECADE_OUTDIR)

    print("\n" + "=" * 60 + "\n")
    print("FINAL SUMMARY")
    print(f"ERA - Downloaded: {len(era_downloaded)}, Skipped: {len(era_skipped)}, Failed: {len(era_failed)}")
    print(f"DECADE - Downloaded: {len(decade_downloaded)}, Skipped: {len(decade_skipped)}, Failed: {len(decade_failed)}")

    print("\nFINAL FAILED IDS")
    print("ERA:", era_failed)
    print("DECADE:", decade_failed)


if __name__ == "__main__":
    main()