import logging
import os
import sys

from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = "app.log"

def init_logging(log_dir: Path = LOG_DIR, log_file: str = LOG_FILE, level: int = logging.INFO) -> None:
    """Initialize logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )