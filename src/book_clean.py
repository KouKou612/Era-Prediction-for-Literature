from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"

ERA_RAW_DIR = DATASET_DIR / "era_sample_raw"
DECADE_RAW_DIR = DATASET_DIR / "decade_sample_raw"

ERA_CLEAN_DIR = DATASET_DIR / "era_sample_clean"
DECADE_CLEAN_DIR = DATASET_DIR / "decade_sample_clean"


START_PATTERNS = [
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
]

END_PATTERNS = [
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
]


def find_earliest_start(text: str) -> int | None:
    earliest = None
    for pattern in START_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            pos = match.end()
            if earliest is None or pos < earliest:
                earliest = pos
    return earliest


def find_latest_end(text: str) -> int | None:
    latest = None
    for pattern in END_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            pos = matches[-1].start()
            if latest is None or pos > latest:
                latest = pos
    return latest


def clean_gutenberg_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start_pos = find_earliest_start(text)
    if start_pos is not None:
        text = text[start_pos:]

    end_pos = find_latest_end(text)
    if end_pos is not None:
        text = text[:end_pos]

    text = text.strip()

    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def clean_directory(raw_dir: Path, clean_dir: Path) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_dir.glob("*.txt"))
    total = len(txt_files)

    if total == 0:
        print(f"No .txt files found in {raw_dir}")
        return

    cleaned = 0
    skipped_empty = 0
    failed = 0

    for i, file_path in enumerate(txt_files, start=1):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_text = clean_gutenberg_text(text)

            if not cleaned_text.strip():
                print(f"[{i}/{total}] Empty after cleaning: {file_path.name}")
                skipped_empty += 1
                continue

            out_path = clean_dir / file_path.name
            out_path.write_text(cleaned_text, encoding="utf-8")

            print(f"[{i}/{total}] Cleaned: {file_path.name}")
            cleaned += 1

        except Exception as e:
            print(f"[{i}/{total}] Failed: {file_path.name} | {e}")
            failed += 1

    print(f"\nFinished cleaning: {raw_dir.name}")
    print(f"Output folder: {clean_dir}")
    print(f"Cleaned: {cleaned}")
    print(f"Empty after cleaning: {skipped_empty}")
    print(f"Failed: {failed}")


def main() -> None:
    print("Cleaning ERA raw texts...\n")
    clean_directory(ERA_RAW_DIR, ERA_CLEAN_DIR)

    print("\n" + "=" * 60 + "\n")

    print("Cleaning DECADE raw texts...\n")
    clean_directory(DECADE_RAW_DIR, DECADE_CLEAN_DIR)


if __name__ == "__main__":
    main()