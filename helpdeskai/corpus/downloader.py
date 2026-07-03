"""Orchestration logic to download and write raw corpus artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import load_dataset

from .config import (
    BITEXT_REPO,
    CHECKSUM_FILE,
    MSDIALOG_URL,
    TECHQA_REPO,
    DownloadConfig,
)
from .datasets import records_from_split, select_split
from .io_utils import sha256, verify_checksums, write_jsonl
from .transforms import map_bitext, map_msdialog, map_techqa_doc, map_techqa_qa


def run_download(config: DownloadConfig) -> None:
    """Download datasets, build subsets, write JSONL artifacts, and checksum them."""

    out_dir: Path = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "techqa_documents.jsonl": out_dir / "techqa_documents.jsonl",
        "techqa_qa.jsonl": out_dir / "techqa_qa.jsonl",
        "bitext_pairs.jsonl": out_dir / "bitext_pairs.jsonl",
        "msdialog_conversations.jsonl": out_dir / "msdialog_conversations.jsonl",
    }

    if (
        not config.overwrite
        and all(path.exists() for path in output_files.values())
        and verify_checksums(out_dir)
    ):
        logging.info("All dataset artifacts already exist and checksum verification passed.")
        return

    logging.info("Loading TechQA from %s", TECHQA_REPO)
    techqa_raw = load_dataset(TECHQA_REPO)

    techqa_docs_split = select_split(techqa_raw, ["corpus", "documents", "doc", "train"])
    techqa_qa_split = select_split(techqa_raw, ["qa", "questions", "validation", "test", "train"])

    logging.info(
        "Sampling %d TechQA documents and %d QA pairs", config.techqa_docs, config.techqa_qa
    )
    techqa_docs = records_from_split(
        techqa_docs_split,
        mapper=map_techqa_doc,
        n_rows=config.techqa_docs,
        seed=config.seed,
    )
    techqa_qa = records_from_split(
        techqa_qa_split,
        mapper=map_techqa_qa,
        n_rows=config.techqa_qa,
        seed=config.seed,
    )

    logging.info("Loading Bitext from %s", BITEXT_REPO)
    bitext_raw = load_dataset(BITEXT_REPO)
    bitext_split = select_split(bitext_raw, ["train", "validation", "test"])
    logging.info("Sampling %d Bitext pairs", config.bitext)
    bitext_pairs = records_from_split(
        bitext_split,
        mapper=map_bitext,
        n_rows=config.bitext,
        seed=config.seed,
    )

    logging.info("Loading MSDialog from JSON URL")
    msdialog_split = load_dataset("json", data_files=MSDIALOG_URL, split="train")
    logging.info("Sampling %d MSDialog conversations", config.msdialog)
    msdialog_rows = records_from_split(
        msdialog_split,
        mapper=map_msdialog,
        n_rows=config.msdialog,
        seed=config.seed,
    )

    write_jsonl(output_files["techqa_documents.jsonl"], techqa_docs)
    write_jsonl(output_files["techqa_qa.jsonl"], techqa_qa)
    write_jsonl(output_files["bitext_pairs.jsonl"], bitext_pairs)
    write_jsonl(output_files["msdialog_conversations.jsonl"], msdialog_rows)

    checksums = {name: sha256(path) for name, path in output_files.items()}
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
