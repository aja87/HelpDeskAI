"""Ingestion utilities for HelpDeskAI phase 2."""

from __future__ import annotations

from typing import Any

__all__ = ["IngestionConfig", "run_ingestion_flow"]


def __getattr__(name: str) -> Any:
	"""Load ingestion pipeline exports lazily."""

	if name in __all__:
		from .pipeline import IngestionConfig, run_ingestion_flow

		exports = {
			"IngestionConfig": IngestionConfig,
			"run_ingestion_flow": run_ingestion_flow,
		}
		return exports[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")