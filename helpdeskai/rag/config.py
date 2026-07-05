from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/rag")
GOLDEN_DIR = Path("tests/golden")

DEFAULT_CHUNKS_PATH = PROCESSED_DIR / "techqa_chunks.jsonl"
DEFAULT_GOLDEN_PATH = GOLDEN_DIR / "golden_dataset.jsonl"
DEFAULT_EVALUATION_PATH = REPORTS_DIR / "evaluation_report.json"

LOG_FILE = "rag.log"

VALID_PROMPT_VARIANTS = {"baseline", "grounded", "concise"}
VALID_RETRIEVAL_MODES = {"dense", "sparse", "hybrid"}


@dataclass(slots=True)
class RagConfig:
    """Runtime configuration for the RAG pipeline."""

    chunks_path: Path = DEFAULT_CHUNKS_PATH
    golden_path: Path = DEFAULT_GOLDEN_PATH
    evaluation_path: Path = DEFAULT_EVALUATION_PATH

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "helpdeskai-techqa"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_api_base: str = "https://api.anthropic.com"
    rewrite_model: str = "claude-haiku-4-5-20251001"
    generator_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "claude-sonnet-4-6"
    mock_llm: bool = True

    retrieval_mode: str = "hybrid"
    prompt_variant: str = "grounded"
    top_k: int = 20
    rerank_top_k: int = 5
    compression_top_k: int = 5
    evaluation_sample_size: int = 25
    max_faithfulness_drop: float = 0.05

    def validate(self, *, require_golden: bool = True) -> None:
        """Validate configuration values before execution."""

        if not self.chunks_path.exists():
            raise FileNotFoundError(f"chunks_path does not exist: {self.chunks_path}")
        if require_golden and not self.golden_path.exists():
            raise FileNotFoundError(f"golden_path does not exist: {self.golden_path}")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if not self.reranker_model.strip():
            raise ValueError("reranker_model must not be empty")
        if not self.rewrite_model.strip():
            raise ValueError("rewrite_model must not be empty")
        if not self.generator_model.strip():
            raise ValueError("generator_model must not be empty")
        if not self.judge_model.strip():
            raise ValueError("judge_model must not be empty")
        if not self.mock_llm and not (self.anthropic_api_key and self.anthropic_api_key.strip()):
            raise ValueError("ANTHROPIC_API_KEY is required when mock_llm is disabled")
        if self.retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {sorted(VALID_RETRIEVAL_MODES)}")
        if self.prompt_variant not in VALID_PROMPT_VARIANTS:
            raise ValueError(f"prompt_variant must be one of {sorted(VALID_PROMPT_VARIANTS)}")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be > 0")
        if self.compression_top_k <= 0:
            raise ValueError("compression_top_k must be > 0")
        if self.evaluation_sample_size <= 0:
            raise ValueError("evaluation_sample_size must be > 0")
        if self.max_faithfulness_drop < 0:
            raise ValueError("max_faithfulness_drop must be >= 0")