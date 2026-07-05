
from dataclasses import dataclass
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/retrieval")
GOLDEN_PATH = Path("tests/golden/golden_dataset.jsonl")

DEFAULT_CHUNKS_PATH = PROCESSED_DIR / "techqa_chunks.jsonl"
DEFAULT_BENCHMARK_PATH = REPORTS_DIR / "benchmark_report.json"

LOG_FILE = "retrieval.log"

VALID_RETRIEVAL_MODES = {"dense", "sparse", "hybrid"}


@dataclass(slots=True)
class RetrievalConfig:
    """Runtime configuration for retrieval search and benchmark."""

    chunks_path: Path = DEFAULT_CHUNKS_PATH
    golden_path: Path = GOLDEN_PATH
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "helpdeskai-techqa"
    embedding_model: str = "BAAI/bge-m3"

    mode: str = "hybrid"
    top_k: int = 5
    rrf_k: int = 60

    benchmark_sample_size: int = 100
    benchmark_relevance_top_n: int = 3

    def validate(self) -> None:
        """Validate configuration values before execution."""

        if not self.chunks_path.exists():
            raise FileNotFoundError(f"chunks_path does not exist: {self.chunks_path}")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_RETRIEVAL_MODES)}")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if self.benchmark_sample_size <= 0:
            raise ValueError("benchmark_sample_size must be > 0")
        if self.benchmark_relevance_top_n <= 0:
            raise ValueError("benchmark_relevance_top_n must be > 0")