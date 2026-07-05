from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


REPORTS_DIR = Path("reports/agents")
DEFAULT_CHUNKS_PATH = Path("data/processed/techqa_chunks.jsonl")
DEFAULT_GRAPH_PATH = REPORTS_DIR / "workflow_graph.mmd"
DEFAULT_CHECKPOINT_PATH = REPORTS_DIR / "agent_checkpoints.sqlite"
DEFAULT_SESSION_ID = "local-session"

DEFAULT_COLLECTION_NAME = "helpdeskai-techqa"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GENERATOR_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"

LOG_FILE = "agents.log"

VALID_INTENTS = {"chitchat", "clarify", "escalate", "factual"}
VALID_CHECKPOINT_BACKENDS = {"sqlite", "postgres"}


@dataclass(slots=True)
class AgentsConfig:
	"""Runtime configuration for the agentic support workflow."""

	chunks_path: Path = DEFAULT_CHUNKS_PATH
	checkpoint_backend: str = "sqlite"
	checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
	checkpoint_dsn: str | None = None
	session_id: str = DEFAULT_SESSION_ID

	qdrant_url: str = "http://localhost:6333"
	qdrant_api_key: str | None = None
	collection_name: str = DEFAULT_COLLECTION_NAME
	embedding_model: str = DEFAULT_EMBEDDING_MODEL

	anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
	anthropic_api_base: str = DEFAULT_ANTHROPIC_API_BASE
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL
	generator_model: str = DEFAULT_GENERATOR_MODEL
	mock_llm: bool = True

	max_iterations: int = 5
	max_tokens: int = 10000
	confidence_threshold: float = 0.65
	top_k: int = 5

	graph_path: Path = DEFAULT_GRAPH_PATH

	def validate(self) -> None:
		"""Validate runtime configuration before workflow execution."""

		if not self.chunks_path.exists():
			raise FileNotFoundError(f"chunks_path does not exist: {self.chunks_path}")
		if self.checkpoint_backend not in VALID_CHECKPOINT_BACKENDS:
			raise ValueError(f"checkpoint_backend must be one of {sorted(VALID_CHECKPOINT_BACKENDS)}")
		if self.checkpoint_backend == "postgres" and not (self.checkpoint_dsn and self.checkpoint_dsn.strip()):
			raise ValueError("checkpoint_dsn is required when checkpoint_backend=postgres")
		if self.max_iterations <= 0:
			raise ValueError("max_iterations must be > 0")
		if self.max_iterations > 5:
			raise ValueError("max_iterations must be <= 5")
		if self.max_tokens <= 0:
			raise ValueError("max_tokens must be > 0")
		if self.max_tokens > 10000:
			raise ValueError("max_tokens must be <= 10000")
		if not self.classifier_model.strip():
			raise ValueError("classifier_model must not be empty")
		if not self.generator_model.strip():
			raise ValueError("generator_model must not be empty")
		if not self.collection_name.strip():
			raise ValueError("collection_name must not be empty")
		if not self.embedding_model.strip():
			raise ValueError("embedding_model must not be empty")
		if not self.mock_llm and not (self.anthropic_api_key and self.anthropic_api_key.strip()):
			raise ValueError("ANTHROPIC_API_KEY is required when mock_llm is disabled")