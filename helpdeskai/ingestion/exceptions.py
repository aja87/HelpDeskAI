"""Ingestion-specific exceptions."""


class TechQAIngestionError(RuntimeError):
    """Raised when raw TechQA data cannot be processed safely."""
