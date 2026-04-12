import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Dataset" / "books_raw"
CLEAN = ROOT / "Dataset" / "books_clean"

CLEAN.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    # remove header
    text = re.split(r"\*\*\* START OF.*\*\*\*", text, maxsplit=1, flags=re.IGNORECASE)[-1]

    # remove footer
    text = re.split(r"\*\*\* END OF.*\*\*\*", text, maxsplit=1, flags=re.IGNORECASE)[0]

    return text.strip()


def main() -> None:
    for path in sorted(RAW.glob("*.txt")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = clean_text(text)

        out = CLEAN / path.name
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)

        print("cleaned", path.name)


if __name__ == "__main__":
    main()
