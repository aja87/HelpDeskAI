"""Phase 2 ingestion pipeline for HelpDeskAI.

The module normalizes the raw corpora produced by ``scripts/download_corpus.py``
and generates the main phase-2 artifacts:

- normalized TechQA, Bitext, and MSDialog JSONL files
- chunked TechQA corpus ready for indexing
- a benchmark report comparing chunking strategies on 50 documents
- an Evidently HTML quality report for the normalized document corpus
- a deterministic 100-question golden dataset in ``tests/golden``

It can run as a plain Python module or as a Prefect flow.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable, TypeVar


try:
    from prefect import flow, task
except ImportError:  # pragma: no cover - exercised only when Prefect is absent.
    F = TypeVar("F", bound=Callable[..., Any])

    def _passthrough_decorator(fn: F | None = None, **_: Any) -> Callable[[F], F] | F:
        if fn is not None:
            return fn

        def wrapper(inner: F) -> F:
            return inner

        return wrapper

    flow = _passthrough_decorator
    task = _passthrough_decorator


try:
    from evidently.legacy.metric_preset import DataQualityPreset
    from evidently.legacy.report import Report
except ImportError:  # pragma: no cover - exercised only when Evidently is absent.
    DataQualityPreset = None
    Report = None


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/ingestion")
GOLDEN_DIR = Path("tests/golden")

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
VERSION_RE = re.compile(r"\b\d+(?:\.\d+){1,3}\b")

CATEGORY_MARKERS = {
    "TROUBLESHOOTING": "troubleshooting",
    "SECURITY BULLETIN": "security_bulletin",
    "TECHNOTE": "technote",
    "DOWNLOAD": "download",
    "FAQ": "faq",
    "INSTALLATION": "installation",
    "CONFIGURATION": "configuration",
}


@dataclass(slots=True)
class IngestionConfig:
    """Runtime configuration for the phase-2 ingestion pipeline."""

    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    reports_dir: Path = REPORTS_DIR
    golden_dir: Path = GOLDEN_DIR
    seed: int = 42
    chunk_sample_size: int = 50
    golden_size: int = 100
    fixed_chunk_size: int = 1200
    fixed_overlap: int = 120
    recursive_chunk_size: int = 1000
    recursive_overlap: int = 120
    semantic_chunk_size: int = 900


def parse_args() -> IngestionConfig:
    """Parse CLI arguments and return the ingestion configuration."""

    parser = argparse.ArgumentParser(description="Run the HelpDeskAI phase-2 ingestion pipeline")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-sample-size", type=int, default=50)
    parser.add_argument("--golden-size", type=int, default=100)
    return IngestionConfig(**vars(parser.parse_args()))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write iterable records to JSONL format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document with deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    """Clean HTML fragments and normalize spacing for downstream indexing."""

    if not text:
        return ""
    cleaned = html.unescape(text)
    cleaned = cleaned.replace("[SEP]", "\n")
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = BLANK_LINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def normalize_label(value: str) -> str:
    """Normalize label-like metadata into lowercase snake case."""

    cleaned = normalize_text(value)
    if not cleaned:
        return ""
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def stable_checksum(text: str) -> str:
    """Build a deterministic checksum for deduplication and provenance."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_title(text: str) -> str:
    """Use the first non-empty line as a fallback title."""

    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:160]
    return ""


def infer_product(text: str) -> str:
    """Extract a rough product name from the first line of TechQA documents."""

    first_line = infer_title(text)
    patterns = [
        r"^IBM\s+(.*?)\s+-\s+United States",
        r"^IBM\s+(.*?)\s+(?:TECHNOTE|SECURITY BULLETIN|FAQ|DOWNLOAD)",
    ]
    for pattern in patterns:
        match = re.search(pattern, first_line, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1))
    return ""


def infer_version(text: str) -> str:
    """Extract a likely version string from a document body."""

    match = VERSION_RE.search(text)
    return match.group(0) if match else ""


def infer_category(text: str) -> str:
    """Infer a coarse category from common markers in TechQA documents."""

    upper_text = text.upper()
    for marker, label in CATEGORY_MARKERS.items():
        if marker in upper_text:
            return label
    return ""


def prepare_techqa_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize TechQA documents and drop exact duplicates."""

    prepared: list[dict[str, Any]] = []
    seen_checksums: set[str] = set()
    for raw in raw_documents:
        text = normalize_text(str(raw.get("text", "")))
        if not text:
            continue
        checksum = stable_checksum(text)
        if checksum in seen_checksums:
            continue
        seen_checksums.add(checksum)

        title = normalize_text(str(raw.get("title", ""))) or infer_title(text)
        product = normalize_text(str(raw.get("product", ""))) or infer_product(text)
        version = normalize_text(str(raw.get("version", ""))) or infer_version(text)
        category = normalize_label(str(raw.get("category", ""))) or infer_category(text)
        date = normalize_text(str(raw.get("date", "")))
        source_doc_id = normalize_text(str(raw.get("doc_id", "")))

        prepared.append(
            {
                "doc_id": source_doc_id or checksum[:16],
                "source_doc_id": source_doc_id,
                "title": title,
                "text": text,
                "product": product,
                "version": version,
                "category": category,
                "date": date,
                "source": "techqa",
                "checksum": checksum,
                "char_count": len(text),
                "word_count": len(text.split()),
            }
        )
    return prepared


def prepare_qa_pairs(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Normalize question-answer corpora into a shared evaluation schema."""

    prepared: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in rows:
        question = normalize_text(str(row.get("question", "")))
        answer = normalize_text(str(row.get("answer", "")))
        if not question or not answer:
            continue

        question_key = question.lower()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)

        explicit_id = row.get("question_id") or row.get("id") or row.get("conversation_id") or ""
        generated_id = f"{source}_{stable_checksum(question)[:12]}"
        prepared.append(
            {
                "question_id": normalize_text(str(explicit_id)) or generated_id,
                "question": question,
                "answer": answer,
                "doc_id": normalize_text(str(row.get("doc_id", ""))),
                "intent": normalize_label(str(row.get("intent", ""))),
                "category": normalize_label(str(row.get("category", ""))),
                "source": source,
            }
        )
    return prepared


def prepare_msdialog_conversations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize MSDialog conversations for future agent evaluation scenarios."""

    prepared: list[dict[str, Any]] = []
    for row in rows:
        text = normalize_text(str(row.get("text", "")))
        if not text:
            continue
        conversation_id = normalize_text(str(row.get("conversation_id", "")))
        prepared.append(
            {
                "conversation_id": conversation_id or stable_checksum(text)[:12],
                "text": text,
                "final_answer": normalize_text(str(row.get("final_answer", ""))),
                "intent": normalize_label(str(row.get("intent", ""))),
                "category": normalize_label(str(row.get("category", ""))),
                "source": "msdialog",
                "checksum": stable_checksum(text),
            }
        )
    return prepared


def _with_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Apply simple character overlap between already packed chunks."""

    if overlap <= 0:
        return chunks
    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue
        prefix = chunks[index - 1][-overlap:].strip()
        merged = f"{prefix}\n{chunk}" if prefix else chunk
        overlapped.append(merged.strip())
    return overlapped


def chunk_fixed_size(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[str]:
    """Chunk text by fixed character windows."""

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _pack_segments(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Pack semantic segments into chunks close to a target size."""

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        segment_size = len(segment)
        if current and current_size + segment_size + 2 > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = [segment]
            current_size = segment_size
            continue
        current.append(segment)
        current_size += segment_size + 2

    if current:
        chunks.append("\n\n".join(current).strip())
    return _with_overlap(chunks, overlap)


def chunk_recursive(text: str, chunk_size: int = 1000, overlap: int = 120) -> list[str]:
    """Chunk by paragraphs first, then by sentence groups for long sections."""

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    expanded_segments: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            expanded_segments.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        expanded_segments.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return _pack_segments(expanded_segments, chunk_size, overlap)


def chunk_semantic(text: str, chunk_size: int = 900) -> list[str]:
    """Approximate semantic chunking by favoring section boundaries and headings."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in lines:
        is_heading = line.isupper() or line.endswith(":") or len(line) < 80
        line_size = len(line)
        if current and (is_heading or current_size + line_size + 1 > chunk_size):
            segments.append("\n".join(current).strip())
            current = [line]
            current_size = line_size
            continue
        current.append(line)
        current_size += line_size + 1

    if current:
        segments.append("\n".join(current).strip())
    return segments


def benchmark_chunking_strategies(
    documents: list[dict[str, Any]], config: IngestionConfig
) -> dict[str, Any]:
    """Compare fixed, recursive, and semantic chunking on a deterministic sample."""

    rng = random.Random(config.seed)
    sample_size = min(config.chunk_sample_size, len(documents))
    sampled_documents = rng.sample(documents, sample_size) if sample_size else []

    strategies: dict[str, Callable[[str], list[str]]] = {
        "fixed": lambda text: chunk_fixed_size(text, config.fixed_chunk_size, config.fixed_overlap),
        "recursive": lambda text: chunk_recursive(
            text, config.recursive_chunk_size, config.recursive_overlap
        ),
        "semantic": lambda text: chunk_semantic(text, config.semantic_chunk_size),
    }

    results: dict[str, Any] = {}
    target = config.recursive_chunk_size
    for name, chunker in strategies.items():
        chunk_lengths: list[int] = []
        chunks_per_doc: list[int] = []
        for document in sampled_documents:
            chunks = [chunk for chunk in chunker(document["text"]) if chunk.strip()]
            if not chunks:
                continue
            chunks_per_doc.append(len(chunks))
            chunk_lengths.extend(len(chunk) for chunk in chunks)

        if not chunk_lengths:
            results[name] = {
                "chunk_count": 0,
                "avg_chunk_chars": 0,
                "median_chunk_chars": 0,
                "avg_chunks_per_doc": 0,
                "oversized_ratio": 0,
                "undersized_ratio": 0,
                "score": 9999,
            }
            continue

        oversized_ratio = sum(length > 1400 for length in chunk_lengths) / len(chunk_lengths)
        undersized_ratio = sum(length < 300 for length in chunk_lengths) / len(chunk_lengths)
        avg_chunk_chars = mean(chunk_lengths)
        score = abs(avg_chunk_chars - target) + 150 * oversized_ratio + 75 * undersized_ratio

        results[name] = {
            "chunk_count": len(chunk_lengths),
            "avg_chunk_chars": round(avg_chunk_chars, 2),
            "median_chunk_chars": round(median(chunk_lengths), 2),
            "avg_chunks_per_doc": round(mean(chunks_per_doc), 2),
            "oversized_ratio": round(oversized_ratio, 4),
            "undersized_ratio": round(undersized_ratio, 4),
            "score": round(score, 2),
        }

    recommended = min(results.items(), key=lambda item: item[1]["score"])[0] if results else "recursive"
    return {
        "sample_size": sample_size,
        "recommended_strategy": recommended,
        "strategies": results,
    }


def chunk_documents(
    documents: list[dict[str, Any]], strategy_name: str, config: IngestionConfig
) -> list[dict[str, Any]]:
    """Create chunk-level records ready for vector indexing."""

    strategies: dict[str, Callable[[str], list[str]]] = {
        "fixed": lambda text: chunk_fixed_size(text, config.fixed_chunk_size, config.fixed_overlap),
        "recursive": lambda text: chunk_recursive(
            text, config.recursive_chunk_size, config.recursive_overlap
        ),
        "semantic": lambda text: chunk_semantic(text, config.semantic_chunk_size),
    }
    chunker = strategies[strategy_name]

    chunk_rows: list[dict[str, Any]] = []
    for document in documents:
        for index, chunk in enumerate(chunker(document["text"])):
            normalized_chunk = normalize_text(chunk)
            if not normalized_chunk:
                continue
            chunk_rows.append(
                {
                    "chunk_id": f"{document['doc_id']}_{index:04d}",
                    "doc_id": document["doc_id"],
                    "chunk_index": index,
                    "strategy": strategy_name,
                    "text": normalized_chunk,
                    "title": document["title"],
                    "product": document["product"],
                    "version": document["version"],
                    "category": document["category"],
                    "date": document["date"],
                    "source": document["source"],
                }
            )
    return chunk_rows


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


def build_golden_dataset(
    techqa_pairs: list[dict[str, Any]],
    bitext_pairs: list[dict[str, Any]],
    output_path: Path,
    target_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create a deterministic evaluation set balanced across TechQA and Bitext."""

    rng = random.Random(seed)
    techqa_target = min(len(techqa_pairs), target_size // 2)
    bitext_target = min(len(bitext_pairs), target_size - techqa_target)

    techqa_pool = techqa_pairs[:]
    bitext_pool = bitext_pairs[:]
    rng.shuffle(techqa_pool)
    rng.shuffle(bitext_pool)

    selected = techqa_pool[:techqa_target] + bitext_pool[:bitext_target]

    if len(selected) < target_size:
        combined = techqa_pool[techqa_target:] + bitext_pool[bitext_target:]
        for candidate in combined:
            if len(selected) >= target_size:
                break
            selected.append(candidate)

    golden_rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in selected:
        question_key = row["question"].lower()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        golden_rows.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "expected_answer": row["answer"],
                "source": row["source"],
                "doc_id": row["doc_id"],
                "intent": row["intent"],
                "category": row["category"],
            }
        )
        if len(golden_rows) >= target_size:
            break

    write_jsonl(output_path, golden_rows)
    return golden_rows


@task
def load_raw_corpora(config: IngestionConfig) -> dict[str, list[dict[str, Any]]]:
    """Load all raw phase-2 corpora from disk."""

    return {
        "techqa_documents": read_jsonl(config.raw_dir / "techqa_documents.jsonl"),
        "techqa_qa": read_jsonl(config.raw_dir / "techqa_qa.jsonl"),
        "bitext_pairs": read_jsonl(config.raw_dir / "bitext_pairs.jsonl"),
        "msdialog": read_jsonl(config.raw_dir / "msdialog_conversations.jsonl"),
    }


@task
def normalize_corpora(raw: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize every corpus into deterministic downstream schemas."""

    return {
        "documents": prepare_techqa_documents(raw["techqa_documents"]),
        "techqa_qa": prepare_qa_pairs(raw["techqa_qa"], source="techqa"),
        "bitext_pairs": prepare_qa_pairs(raw["bitext_pairs"], source="bitext"),
        "msdialog": prepare_msdialog_conversations(raw["msdialog"]),
    }


@task
def persist_outputs(
    normalized: dict[str, list[dict[str, Any]]],
    chunk_rows: list[dict[str, Any]],
    chunk_report: dict[str, Any],
    quality_summary: dict[str, Any],
    config: IngestionConfig,
) -> dict[str, str]:
    """Persist normalized corpora, chunks, summaries, and manifest files."""

    processed_dir = config.processed_dir
    reports_dir = config.reports_dir
    golden_dir = config.golden_dir

    normalized_documents_path = processed_dir / "techqa_documents_normalized.jsonl"
    normalized_techqa_qa_path = processed_dir / "techqa_qa_normalized.jsonl"
    normalized_bitext_path = processed_dir / "bitext_pairs_normalized.jsonl"
    normalized_msdialog_path = processed_dir / "msdialog_conversations_normalized.jsonl"
    chunk_path = processed_dir / "techqa_chunks.jsonl"
    chunk_report_path = reports_dir / "chunking_benchmark.json"
    quality_summary_path = reports_dir / "quality_summary.json"
    manifest_path = processed_dir / "ingestion_manifest.json"
    golden_path = golden_dir / "golden_dataset.jsonl"

    write_jsonl(normalized_documents_path, normalized["documents"])
    write_jsonl(normalized_techqa_qa_path, normalized["techqa_qa"])
    write_jsonl(normalized_bitext_path, normalized["bitext_pairs"])
    write_jsonl(normalized_msdialog_path, normalized["msdialog"])
    write_jsonl(chunk_path, chunk_rows)
    write_json(chunk_report_path, chunk_report)
    write_json(quality_summary_path, quality_summary)

    build_golden_dataset(
        normalized["techqa_qa"],
        normalized["bitext_pairs"],
        golden_path,
        target_size=config.golden_size,
        seed=config.seed,
    )

    manifest = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "artifacts": {
            "documents": str(normalized_documents_path),
            "techqa_qa": str(normalized_techqa_qa_path),
            "bitext_pairs": str(normalized_bitext_path),
            "msdialog": str(normalized_msdialog_path),
            "chunks": str(chunk_path),
            "chunk_benchmark": str(chunk_report_path),
            "quality_summary": str(quality_summary_path),
            "golden_dataset": str(golden_path),
        },
        "counts": {
            "documents": len(normalized["documents"]),
            "techqa_qa": len(normalized["techqa_qa"]),
            "bitext_pairs": len(normalized["bitext_pairs"]),
            "msdialog": len(normalized["msdialog"]),
            "chunks": len(chunk_rows),
        },
    }
    write_json(manifest_path, manifest)

    return {"manifest": str(manifest_path), "golden_dataset": str(golden_path)}


def _run_ingestion_core(config: IngestionConfig) -> dict[str, str]:
    """Execute the ingestion pipeline without requiring a Prefect runtime."""

    raw = load_raw_corpora.fn(config) if hasattr(load_raw_corpora, "fn") else load_raw_corpora(config)
    normalized = (
        normalize_corpora.fn(raw) if hasattr(normalize_corpora, "fn") else normalize_corpora(raw)
    )
    chunk_report = benchmark_chunking_strategies(normalized["documents"], config)
    strategy_name = chunk_report["recommended_strategy"]
    chunk_rows = chunk_documents(normalized["documents"], strategy_name, config)
    quality_summary = build_quality_report(
        normalized["documents"], config.reports_dir / "data_quality.html"
    )
    if hasattr(persist_outputs, "fn"):
        return persist_outputs.fn(normalized, chunk_rows, chunk_report, quality_summary, config)
    return persist_outputs(normalized, chunk_rows, chunk_report, quality_summary, config)


@flow(name="helpdeskai-ingestion")
def run_ingestion_flow(config: IngestionConfig | None = None) -> dict[str, str]:
    """Run the full phase-2 ingestion pipeline."""

    config = config or IngestionConfig()
    return _run_ingestion_core(config)


def main() -> None:
    """CLI entrypoint for local ingestion runs."""

    _run_ingestion_core(parse_args())


if __name__ == "__main__":
    main()
