from loguru import logger
import os

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

logger.add(
    f"{LOG_DIR}/crimeguard.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO",
    enqueue=True
)

def get_logger():
    return logger