from pathlib import Path
import os

def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def create_project_directories():

    folders = [
        "models/checkpoints",
        "outputs",
        "logs",
        "data/raw",
        "data/processed"
    ]

    for folder in folders:
        ensure_directory(folder)