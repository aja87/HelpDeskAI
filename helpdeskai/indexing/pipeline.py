"""Compatibility facade for indexing utilities."""

from .config import IndexingConfig
from .workflow import run_indexing_core

__all__ = ["IndexingConfig", "run_indexing_core"]
