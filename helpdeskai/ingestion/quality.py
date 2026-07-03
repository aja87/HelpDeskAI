"""Quality metrics and report generation for normalized corpora."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


try:
    from evidently.legacy.metric_preset import DataQualityPreset
    from evidently.legacy.report import Report
except ImportError:  # pragma: no cover - exercised only when Evidently is absent.
    DataQualityPreset = None
    Report = None


def build_quality_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic summary metrics for the normalized corpus."""

    if not documents:
        return {
            "document_count": 0,
            "duplicate_doc_ids": 0,
            "missing_product_ratio": 0,
            "missing_version_ratio": 0,
            "missing_category_ratio": 0,
            "avg_char_count": 0,
            "avg_word_count": 0,
        }

    total = len(documents)
    doc_ids = [doc["doc_id"] for doc in documents]
    duplicate_doc_ids = len(doc_ids) - len(set(doc_ids))
    return {
        "document_count": total,
        "duplicate_doc_ids": duplicate_doc_ids,
        "missing_product_ratio": round(sum(not doc["product"] for doc in documents) / total, 4),
        "missing_version_ratio": round(sum(not doc["version"] for doc in documents) / total, 4),
        "missing_category_ratio": round(sum(not doc["category"] for doc in documents) / total, 4),
        "avg_char_count": round(mean(doc["char_count"] for doc in documents), 2),
        "avg_word_count": round(mean(doc["word_count"] for doc in documents), 2),
    }


def build_quality_report(documents: list[dict[str, Any]], report_path: Path) -> dict[str, Any]:
    """Generate an Evidently report and return a JSON-friendly quality summary."""

    summary = build_quality_summary(documents)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    dataframe = pd.DataFrame(documents)
    dataframe["missing_product"] = dataframe["product"].eq("")
    dataframe["missing_version"] = dataframe["version"].eq("")
    dataframe["missing_category"] = dataframe["category"].eq("")

    try:
        report = Report(metrics=[DataQualityPreset()])
        report.run(current_data=dataframe, reference_data=None)
        report.save_html(str(report_path))
        summary["report_backend"] = "evidently"
    except Exception as exc:  # pragma: no cover - only used on unsupported Evidently APIs.
        report_path.write_text(
            "<html><body><h1>Quality report fallback</h1><pre>"
            + html.escape(json.dumps(summary, indent=2))
            + "\n\nEvidently error: "
            + html.escape(str(exc))
            + "</pre></body></html>",
            encoding="utf-8",
        )
        summary["report_backend"] = "fallback"
        summary["report_error"] = str(exc)

    return summary
