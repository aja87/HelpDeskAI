"""Golden dataset creation helpers for deterministic ingestion outputs."""

from __future__ import annotations

import random
from pathlib import Path

from .io_utils import write_jsonl


def build_golden_dataset(
    techqa_pairs: list[dict[str, str]],
    bitext_pairs: list[dict[str, str]],
    output_path: Path,
    target_size: int,
    seed: int,
) -> list[dict[str, str]]:
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

    golden_rows: list[dict[str, str]] = []
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
