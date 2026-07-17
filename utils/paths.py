from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT_DIR / "data" / "raw"

PROCESSED_DATA = ROOT_DIR / "data" / "processed"

MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR = ROOT_DIR / "outputs"

LOG_DIR = ROOT_DIR / "logs"

for folder in [
    RAW_DATA,
    PROCESSED_DATA,
    MODEL_DIR,
    OUTPUT_DIR,
    LOG_DIR
]:
    folder.mkdir(exist_ok=True)