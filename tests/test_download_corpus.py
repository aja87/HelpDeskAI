from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import download_corpus


def make_techqa_record(split: str, index: int) -> dict[str, str]:
    return {
        "id": f"{split}-{index}",
        "document": f"Document {split} {index}",
        "question": f"Question {index}?",
        "answer": f"Answer {index}",
    }


@pytest.fixture
def fake_dataset_loader():
    def loader(name: str, *, split: str, revision: str):
        if name == download_corpus.TECHQA_QA_DATASET:
            assert revision == download_corpus.TECHQA_QA_REVISION
            size = download_corpus.TECHQA_SPLIT_SIZES[split]
            return [make_techqa_record(split, index) for index in range(size)]
        if name == download_corpus.BITEXT_DATASET:
            assert revision == download_corpus.BITEXT_REVISION
            return [
                {
                    "flags": "B",
                    "instruction": f"Instruction {index}",
                    "category": "ACCOUNT",
                    "intent": "edit_account",
                    "response": f"Response {index}",
                }
                for index in range(2_100)
            ]
        raise AssertionError(f"Unexpected dataset: {name}")

    return loader


@pytest.fixture
def fake_msdialog_bytes() -> bytes:
    return b"".join(
        json.dumps({"conversation_id": index, "utterances": [f"Message {index}"]}).encode() + b"\n"
        for index in range(600)
    )


def fake_document_loader(seed: int, archive_fetcher) -> tuple[list[dict], str]:
    del archive_fetcher
    documents = [
        {
            "id": f"technote-{index:05d}",
            "split": "technotes",
            "document": f"TechNote document {index}",
        }
        for index in range(download_corpus.TECHQA_DOCUMENT_SAMPLE_SIZE)
    ]
    random_order = list(documents)
    import random

    random.Random(seed).shuffle(random_order)
    return random_order, "a" * 64


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            b'[{"text":"first"},{"document":"second"}]',
            [("0", {"text": "first"}), ("1", {"document": "second"})],
        ),
        (
            b'{"doc-1":"first","doc-2":{"content":"second"}}',
            [("doc-1", "first"), ("doc-2", {"content": "second"})],
        ),
    ],
)
def test_streaming_json_parser(payload: bytes, expected: list[tuple[str, object]]) -> None:
    assert list(download_corpus._iter_top_level_json(io.BytesIO(payload))) == expected


def test_download_corpora_writes_expected_outputs(
    tmp_path: Path, fake_dataset_loader, fake_msdialog_bytes: bytes
) -> None:
    data_dir = tmp_path / "raw"

    manifest_path = download_corpus.download_corpora(
        data_dir,
        dataset_loader=fake_dataset_loader,
        url_fetcher=lambda _: fake_msdialog_bytes,
        document_loader=fake_document_loader,
    )

    qa = read_jsonl(data_dir / "techqa/qa.jsonl")
    documents = read_jsonl(data_dir / "techqa/documents.jsonl")
    bitext = read_jsonl(data_dir / "bitext/tickets.jsonl")
    msdialog = read_jsonl(data_dir / "msdialog/conversations.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(qa) == 621
    assert len(documents) == 5_000
    assert len(bitext) == 2_000
    assert len(msdialog) == 500
    assert set(qa[0]) == {"id", "question", "answer", "split"}
    assert set(documents[0]) == {"id", "document", "split"}
    assert manifest["outputs"]["techqa_qa"]["records"] == 621
    assert manifest["outputs"]["bitext"]["records"] == 2_000
    assert manifest["outputs"]["msdialog"]["records"] == 500
    assert (
        manifest["sources"]["msdialog"]["source_sha256"]
        == hashlib.sha256(fake_msdialog_bytes).hexdigest()
    )


def test_sampling_is_reproducible(
    tmp_path: Path, fake_dataset_loader, fake_msdialog_bytes: bytes
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    for data_dir in (first_dir, second_dir):
        download_corpus.download_corpora(
            data_dir,
            seed=123,
            dataset_loader=fake_dataset_loader,
            url_fetcher=lambda _: fake_msdialog_bytes,
            document_loader=fake_document_loader,
        )

    assert (first_dir / "bitext/tickets.jsonl").read_bytes() == (
        second_dir / "bitext/tickets.jsonl"
    ).read_bytes()
    assert (first_dir / "msdialog/conversations.jsonl").read_bytes() == (
        second_dir / "msdialog/conversations.jsonl"
    ).read_bytes()


def test_manifest_checksums_match_output_files(
    tmp_path: Path, fake_dataset_loader, fake_msdialog_bytes: bytes
) -> None:
    data_dir = tmp_path / "raw"
    manifest_path = download_corpus.download_corpora(
        data_dir,
        dataset_loader=fake_dataset_loader,
        url_fetcher=lambda _: fake_msdialog_bytes,
        document_loader=fake_document_loader,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for output in manifest["outputs"].values():
        output_path = data_dir / output["path"]
        assert download_corpus.sha256_file(output_path) == output["sha256"]
        assert output_path.stat().st_size == output["bytes"]


def test_existing_outputs_require_force_or_skip(
    tmp_path: Path, fake_dataset_loader, fake_msdialog_bytes: bytes
) -> None:
    data_dir = tmp_path / "raw"
    kwargs = {
        "dataset_loader": fake_dataset_loader,
        "url_fetcher": lambda _: fake_msdialog_bytes,
        "document_loader": fake_document_loader,
    }
    manifest_path = download_corpus.download_corpora(data_dir, **kwargs)

    with pytest.raises(download_corpus.CorpusDownloadError, match="--force"):
        download_corpus.download_corpora(data_dir, **kwargs)

    assert download_corpus.download_corpora(data_dir, skip_existing=True, **kwargs) == manifest_path

    download_corpus.download_corpora(data_dir, seed=99, force=True, **kwargs)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["seed"] == 99


def test_partial_existing_outputs_cannot_be_skipped(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    partial = data_dir / "techqa/qa.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text("{}\n", encoding="utf-8")

    with pytest.raises(download_corpus.CorpusDownloadError, match="--force"):
        download_corpus.download_corpora(
            data_dir,
            skip_existing=True,
            dataset_loader=lambda *args, **kwargs: [],
            url_fetcher=lambda _: b"",
        )


def test_invalid_source_data_is_reported(fake_dataset_loader) -> None:
    invalid_jsonl = b'{"valid": true}\nnot-json\n'

    with pytest.raises(download_corpus.CorpusDownloadError, match="line 2"):
        download_corpus.load_msdialog(42, lambda _: invalid_jsonl)

    def failing_loader(*args, **kwargs):
        raise OSError("network unavailable")

    with pytest.raises(download_corpus.CorpusDownloadError, match="network unavailable"):
        download_corpus.load_bitext(42, failing_loader)


def test_cli_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        download_corpus.parse_args(["--force", "--skip-existing"])
