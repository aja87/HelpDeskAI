"""Download and prepare the datasets used by HelpDeskAI.

The checksum-verified implementation is intentionally left for the ingestion
phase of the project.
"""

from pathlib import Path


DATA_DIR = Path("data/raw")


def main() -> None:
    """Create the target directory before corpus download is implemented."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Implement checksum-verified downloads for TechQA, Bitext, and MSDialog."
    )


if __name__ == "__main__":
    main()

