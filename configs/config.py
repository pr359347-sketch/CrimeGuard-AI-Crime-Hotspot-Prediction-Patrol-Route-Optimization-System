from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_FILE = BASE_DIR / "configs" / "settings.yaml"

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

PROJECT = config["project"]
PATHS = config["paths"]
CRIME = config["crime"]
WEATHER = config["weather"]
TRAINING = config["training"]
MODEL = config["model"]