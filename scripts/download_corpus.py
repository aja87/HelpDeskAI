"""Download and subset the corpora used by HelpDeskAI phase 2.

This script fetches datasets with Hugging Face ``load_dataset`` and writes
reproducible JSONL artifacts in ``data/raw`` along with a checksum manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset


DATA_DIR = Path("data/raw")
CHECKSUM_FILE = "checksums.sha256.json"

TECHQA_REPO = "rojagtap/tech-qa"
BITEXT_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MSDIALOG_URL = (
    "https://raw.githubusercontent.com/SCU-ChenYue/MSDialog_RL/main/test_MSDialog.jsonl"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options controlling output paths and subset sizes."""
    parser = argparse.ArgumentParser(description="Download and subset HelpDeskAI corpora")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--techqa-docs", type=int, default=5000)
    parser.add_argument("--techqa-qa", type=int, default=600)
    parser.add_argument("--bitext", type=int, default=2000)
    parser.add_argument("--msdialog", type=int, default=500)
    return parser.parse_args()


def _as_text(value: Any) -> str:
    """Normalize heterogeneous dataset values into a plain text string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_as_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _pick_first(row: dict[str, Any], candidates: list[str]) -> Any:
    """Return the first non-empty value found in ``row`` for candidate keys."""
    for key in candidates:
        if key in row and row[key] not in (None, "", []):
            return row[key]
    return None


def _select_split(data: DatasetDict | Dataset | IterableDataset, preferred: list[str]) -> Dataset:
    """Select the most suitable split from a dataset object.

    The function prefers split names listed in ``preferred`` and falls back to
    the first available split when no preferred split is present.
    """
    if isinstance(data, Dataset):
        return data
    if isinstance(data, IterableDataset):
        # Convert streaming-like datasets into an in-memory Dataset for sampling.
        return Dataset.from_list(list(data))
    for split in preferred:
        if split in data:
            return data[split]
    first_split = next(iter(data.keys()))
    return data[first_split]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records to JSONL with one object per line."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        # Read in chunks to handle large files without loading everything in memory.
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(base_dir: Path) -> bool:
    """Validate all expected artifact hashes from checksum manifest."""
    checksum_path = base_dir / CHECKSUM_FILE
    if not checksum_path.exists():
        return False
    expected = json.loads(checksum_path.read_text(encoding="utf-8"))
    for filename, expected_hash in expected.items():
        file_path = base_dir / filename
        if not file_path.exists():
            return False
        if file_path.stat().st_size == 0:
            return False
        if _sha256(file_path) != expected_hash:
            return False
    return True


def _records_from_split(
    split: Dataset, mapper: Any, n_rows: int, seed: int
) -> list[dict[str, Any]]:
    """Sample and map rows until ``n_rows`` valid records are collected."""
    rows: list[dict[str, Any]] = []
    sampled = split.shuffle(seed=seed)
    for row in sampled:
        mapped = mapper(row)
        if mapped is None:
            continue
        rows.append(mapped)
        if len(rows) >= n_rows:
            break
    return rows


def _map_techqa_doc(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw TechQA document row to the normalized output schema."""
    text = _as_text(
        _pick_first(
            row,
            [
                "document",
                "doc",
                "text",
                "content",
                "html",
                "body",
                "passage",
            ],
        )
    ).strip()
    if not text:
        return None
    doc_id = _as_text(_pick_first(row, ["id", "doc_id", "document_id", "uid"]))
    if not doc_id:
        # Generate a stable fallback id when source id is missing.
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    return {
        "doc_id": doc_id,
        "text": text,
        "title": _as_text(_pick_first(row, ["title", "subject"])),
        "product": _as_text(_pick_first(row, ["product", "product_name"])),
        "version": _as_text(_pick_first(row, ["version", "product_version"])),
        "category": _as_text(_pick_first(row, ["category", "topic", "type"])),
        "date": _as_text(_pick_first(row, ["date", "published", "updated_at"])),
        "source": "techqa",
    }


def _map_techqa_qa(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw TechQA QA row to a compact evaluation-friendly schema."""
    question = _as_text(_pick_first(row, ["question", "query", "title"])).strip()
    answer = _as_text(_pick_first(row, ["answer", "answers", "response", "label"])).strip()
    if not question or not answer:
        return None
    return {
        "question_id": _as_text(_pick_first(row, ["qid", "id", "question_id"])),
        "question": question,
        "answer": answer,
        "doc_id": _as_text(_pick_first(row, ["doc_id", "document_id", "docno"])),
        "source": "techqa",
    }


def _map_bitext(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw Bitext row to a unified question/answer schema."""
    question = _as_text(
        _pick_first(
            row,
            [
                "instruction",
                "question",
                "utterance",
                "input",
                "customer_question",
                "text",
            ],
        )
    ).strip()
    answer = _as_text(
        _pick_first(row, ["response", "answer", "output", "agent_response", "label"])
    ).strip()
    if not question or not answer:
        return None
    return {
        "id": _as_text(_pick_first(row, ["id", "uid"])),
        "question": question,
        "answer": answer,
        "intent": _as_text(_pick_first(row, ["intent", "intent_label"])),
        "category": _as_text(_pick_first(row, ["category", "topic"])),
        "source": "bitext",
    }


def _map_msdialog(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw MSDialog row to a normalized conversation schema."""
    conversation = _pick_first(
        row,
        [
            "conversations",
            "utterances",
            "dialog",
            "messages",
            "conversation",
            "turns",
            "text",
        ],
    )
    text = _as_text(conversation).strip()
    if not text:
        return None
    return {
        "conversation_id": _as_text(_pick_first(row, ["id", "conversation_id", "uid"])),
        "text": text,
        "final_answer": _as_text(_pick_first(row, ["final_answer", "answer", "response"])),
        "intent": _as_text(_pick_first(row, ["intent", "dialog_act", "label"])),
        "category": _as_text(_pick_first(row, ["category", "topic", "domain"])),
        "source": "msdialog",
    }


def main() -> None:
    """Download datasets, build subsets, write JSONL artifacts, and checksum them."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "techqa_documents.jsonl": out_dir / "techqa_documents.jsonl",
        "techqa_qa.jsonl": out_dir / "techqa_qa.jsonl",
        "bitext_pairs.jsonl": out_dir / "bitext_pairs.jsonl",
        "msdialog_conversations.jsonl": out_dir / "msdialog_conversations.jsonl",
    }

    # Fast path: skip download when existing artifacts are complete and verified.
    if not args.overwrite and all(p.exists() for p in output_files.values()) and _verify_checksums(out_dir):
        logging.info("All dataset artifacts already exist and checksum verification passed.")
        return

    logging.info("Loading TechQA from %s", TECHQA_REPO)
    techqa_raw = load_dataset(TECHQA_REPO)
    # TechQA may expose different split names depending on dataset revision.
    techqa_docs_split = _select_split(techqa_raw, ["corpus", "documents", "doc", "train"])
    techqa_qa_split = _select_split(techqa_raw, ["qa", "questions", "validation", "test", "train"])

    techqa_docs = _records_from_split(
        techqa_docs_split,
        mapper=_map_techqa_doc,
        n_rows=args.techqa_docs,
        seed=args.seed,
    )
    techqa_qa = _records_from_split(
        techqa_qa_split,
        mapper=_map_techqa_qa,
        n_rows=args.techqa_qa,
        seed=args.seed,
    )

    logging.info("Loading Bitext from %s", BITEXT_REPO)
    bitext_raw = load_dataset(BITEXT_REPO)
    bitext_split = _select_split(bitext_raw, ["train", "validation", "test"])
    bitext_pairs = _records_from_split(
        bitext_split,
        mapper=_map_bitext,
        n_rows=args.bitext,
        seed=args.seed,
    )

    logging.info("Loading MSDialog from JSON URL")
    msdialog_split = load_dataset("json", data_files=MSDIALOG_URL, split="train")
    msdialog_rows = _records_from_split(
        msdialog_split,
        mapper=_map_msdialog,
        n_rows=args.msdialog,
        seed=args.seed,
    )

    _write_jsonl(output_files["techqa_documents.jsonl"], techqa_docs)
    _write_jsonl(output_files["techqa_qa.jsonl"], techqa_qa)
    _write_jsonl(output_files["bitext_pairs.jsonl"], bitext_pairs)
    _write_jsonl(output_files["msdialog_conversations.jsonl"], msdialog_rows)

    # Store output hashes so future runs can validate artifact integrity quickly.
    checksums = {name: _sha256(path) for name, path in output_files.items()}
    (out_dir / CHECKSUM_FILE).write_text(
        json.dumps(checksums, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    logging.info(
        "Wrote corpora: techqa_docs=%d techqa_qa=%d bitext=%d msdialog=%d",
        len(techqa_docs),
        len(techqa_qa),
        len(bitext_pairs),
        len(msdialog_rows),
    )
    logging.info("Checksum manifest: %s", out_dir / CHECKSUM_FILE)


if __name__ == "__main__":
    main()

