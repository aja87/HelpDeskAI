"""Dataset loading and sampling utilities for corpus download."""

from __future__ import annotations

import logging
from typing import Any, Callable

from datasets import Dataset, DatasetDict, IterableDataset


def select_split(data: DatasetDict | Dataset | IterableDataset, preferred: list[str]) -> Dataset:
    """Select the most suitable split from a dataset object."""

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


def records_from_split(
    split: Dataset,
    mapper: Callable[[dict[str, Any]], dict[str, Any] | None],
    n_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample and map rows until n_rows valid records are collected."""

    logging.info("Sampling %d rows from split with seed=%d", n_rows, seed)
    rows: list[dict[str, Any]] = []
    sampled = split.shuffle(seed=seed)
    logging.info("Shuffled split contains %d rows", len(sampled))
    for row in sampled:
        mapped = mapper(row)
        if mapped is None:
            continue
        rows.append(mapped)
        if len(rows) >= n_rows:
            break
    return rows
