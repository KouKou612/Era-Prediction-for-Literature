from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "Dataset"

ERA_RAW_DIR = DATASET_DIR / "era_sample_raw"
DECADE_RAW_DIR = DATASET_DIR / "decade_sample_raw"

ERA_CLEAN_DIR = DATASET_DIR / "era_sample_clean"
DECADE_CLEAN_DIR = DATASET_DIR / "decade_sample_clean"

MIN_WORDS = 300
MIN_NONEMPTY_LINES_FOR_RATIO = 10
APPARATUS_LINE_RATIO = 0.28

RE_LINE_MIDI = re.compile(r"(?i)\.mid\b|\.midi\b|\bmidi\b")
RE_LINE_COPY_MARK = re.compile(
    r"(?i)©|\(c\)\s*[, ]?\s*\d{4}|\(p\)\s*[, ]?\s*\d{4}|\ball rights reserved\b|\bcopyrighted\b",
)
RE_LINE_PRODUCED = re.compile(r"(?i)^produced by\s+\S")
RE_LINE_SOUNDTRACK = re.compile(r"(?i)accompanying files contain|soundtrack|\.wav\b|\.mp3\b")

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


def quality_skip_reason(cleaned):
    t = cleaned.strip()
    if not t:
        return "empty"

    words = len(t.split())
    if words < MIN_WORDS:
        return f"too_few_words({words}<{MIN_WORDS})"

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    n_lines = len(lines)
    if n_lines < MIN_NONEMPTY_LINES_FOR_RATIO:
        return None

    bad = 0
    for ln in lines:
        if (
            RE_LINE_MIDI.search(ln)
            or RE_LINE_COPY_MARK.search(ln)
            or RE_LINE_PRODUCED.search(ln)
            or RE_LINE_SOUNDTRACK.search(ln)
        ):
            bad += 1
    ratio = bad / n_lines
    if ratio >= APPARATUS_LINE_RATIO:
        return f"apparatus_lines({bad}/{n_lines}={ratio:.2f}>={APPARATUS_LINE_RATIO})"
    return None


def clean_gutenberg_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    earliest = None
    for pattern in START_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            pos = m.end()
            if earliest is None or pos < earliest:
                earliest = pos
    if earliest is not None:
        text = text[earliest:]

    latest = None
    for pattern in END_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            pos = matches[-1].start()
            if latest is None or pos > latest:
                latest = pos
    if latest is not None:
        text = text[:latest]

    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_directory(raw_dir, clean_dir):
    clean_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_dir.glob("*.txt"))
    total = len(txt_files)

    if total == 0:
        print(f"No .txt files found in {raw_dir}")
        return

    cleaned = 0
    skipped_empty = 0
    skipped_quality = 0
    failed = 0

    for i, file_path in enumerate(txt_files, start=1):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_text = clean_gutenberg_text(text)

            if not cleaned_text.strip():
                print(f"[{i}/{total}] Empty after cleaning: {file_path.name}")
                skipped_empty += 1
                out_path = clean_dir / file_path.name
                if out_path.exists():
                    out_path.unlink()
                continue

            out_path = clean_dir / file_path.name
            qreason = quality_skip_reason(cleaned_text)
            if qreason is not None:
                if out_path.exists():
                    out_path.unlink()
                print(f"[{i}/{total}] Skipped (quality): {file_path.name} | {qreason}")
                skipped_quality += 1
                continue

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
    print(f"Skipped (quality): {skipped_quality}")
    print(f"Failed: {failed}")


def main():
    print("Cleaning ERA raw texts...\n")
    clean_directory(ERA_RAW_DIR, ERA_CLEAN_DIR)

    print("\n" + "=" * 60 + "\n")

    print("Cleaning DECADE raw texts...\n")
    clean_directory(DECADE_RAW_DIR, DECADE_CLEAN_DIR)


if __name__ == "__main__":
    main()
