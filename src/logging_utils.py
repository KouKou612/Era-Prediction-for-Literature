from pathlib import Path
import sys
from datetime import datetime


class Tee:
    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def start_logging(script_name: str, log_dir: Path | None = None) -> Path:
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent

    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{script_name}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    sys.stdout = Tee(log_path)

    print(f"Logging to: {log_path}")
    return log_path