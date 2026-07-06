"""Benchmark dense, sparse, and hybrid retrieval on the golden dataset."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.ingestion.io import read_jsonl  # noqa: E402
from helpdeskai.retrieval.models import RetrievalConfig, SearchMode, SearchResult  # noqa: E402
from helpdeskai.retrieval.search import SearchEngine  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-path", type=Path, default=Path("tests/golden/questions.jsonl"))
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/processed/techqa/chunks.jsonl"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports/retrieval"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def load_cases(path: Path) -> list[dict]:
    cases = []
    for record in read_jsonl(path):
        if record.get("retrieval_eligible") is False:
            continue
        if "document_id" not in record:
            continue
        cases.append(record)
    return cases


def analyze_alignment(cases: Sequence[dict], indexed_document_ids: set[str]) -> dict:
    """Return coverage between golden expected documents and indexed documents."""
    expected_document_ids = {str(case["document_id"]) for case in cases}
    aligned_cases = [
        case for case in cases if str(case["document_id"]) in indexed_document_ids
    ]
    missing = sorted(expected_document_ids - indexed_document_ids)
    return {
        "golden_cases": len(cases),
        "golden_unique_documents": len(expected_document_ids),
        "indexed_unique_documents": len(indexed_document_ids),
        "aligned_cases": len(aligned_cases),
        "aligned_unique_documents": len(
            {str(case["document_id"]) for case in aligned_cases}
        ),
        "missing_unique_documents": len(missing),
        "missing_sample": missing[:10],
    }


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((p / 100) * (len(ordered) - 1))
    return ordered[index]


def benchmark_mode(
    mode: SearchMode,
    cases: Sequence[dict],
    search_fn: Callable[[str, SearchMode], list[SearchResult]],
) -> dict[str, float | str]:
    recalls5 = []
    recalls10 = []
    rrs = []
    latencies = []

    for case in cases:
        relevant = {str(case["document_id"])}
        start = time.perf_counter()
        results = search_fn(str(case["question"]), mode)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved = [result.document_id for result in results]
        recalls5.append(recall_at_k(retrieved, relevant, 5))
        recalls10.append(recall_at_k(retrieved, relevant, 10))
        rrs.append(reciprocal_rank(retrieved, relevant))

    return {
        "mode": mode.value,
        "recall@5": round(statistics.mean(recalls5), 4),
        "recall@10": round(statistics.mean(recalls10), 4),
        "mrr": round(statistics.mean(rrs), 4),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
    }


def write_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def skipped_rows() -> list[dict[str, float | str]]:
    """Return placeholder metric rows when the golden set is not aligned."""
    return [
        {
            "mode": mode.value,
            "recall@5": "n/a",
            "recall@10": "n/a",
            "mrr": "n/a",
            "p50_ms": "n/a",
            "p95_ms": "n/a",
        }
        for mode in SearchMode
    ]


def write_markdown(
    path: Path,
    rows: Sequence[dict[str, float | str]],
    cases: int,
    alignment: dict,
) -> None:
    lines = [
        "# Retrieval Benchmark",
        "",
        f"Retrieval-eligible golden cases: {cases}",
        f"Aligned cases used for metrics: {alignment['aligned_cases']}",
        f"Golden unique documents: {alignment['golden_unique_documents']}",
        f"Indexed unique documents: {alignment['indexed_unique_documents']}",
        f"Missing golden documents from indexed corpus: {alignment['missing_unique_documents']}",
        "",
        "Embedding model: `BAAI/bge-m3`.",
        "",
    ]
    if alignment["aligned_cases"] == 0:
        lines.extend(
            [
                "## Diagnosis",
                "",
                (
                    "The current golden retrieval cases reference TechQA Q/A "
                    "`document_id` values that are not present in the indexed "
                    "TechQA document corpus. Recall metrics are therefore not "
                    "computed for this report."
                ),
                "",
                "Missing document sample:",
                "",
                *[f"- `{document_id}`" for document_id in alignment["missing_sample"]],
                "",
            ]
        )
    elif alignment["missing_unique_documents"]:
        lines.extend(
            [
                "## Alignment Note",
                "",
                (
                    "Some golden documents are missing from the indexed corpus. "
                    "Metrics below use only aligned cases."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "| Mode | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['recall@5']} | {row['recall@10']} | "
            f"{row['mrr']} | {row['p50_ms']} | {row['p95_ms']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.golden_path)
    if not cases:
        print("error: no retrieval-eligible golden cases found")
        return 1

    config = RetrievalConfig(corpus_path=args.corpus_path)
    engine = SearchEngine(config=config)
    indexed_document_ids = {record.document_id for record in engine.records}
    alignment = analyze_alignment(cases, indexed_document_ids)
    aligned_cases = [
        case for case in cases if str(case["document_id"]) in indexed_document_ids
    ]

    def run(query: str, mode: SearchMode) -> list[SearchResult]:
        return engine.search(query, top_k=args.top_k, mode=mode)

    rows = (
        [benchmark_mode(mode, aligned_cases, run) for mode in SearchMode]
        if aligned_cases
        else skipped_rows()
    )
    write_csv(args.report_dir / "benchmark_results.csv", rows)
    write_markdown(args.report_dir / "benchmark_report.md", rows, len(cases), alignment)
    print(f"Wrote retrieval benchmark to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
