"""Download reproducible subsets of the HelpDeskAI source corpora."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from huggingface_hub import hf_hub_download

TECHQA_QA_DATASET = "rojagtap/tech-qa"
TECHQA_DOCUMENT_REPO = "PrimeQA/TechQA"
TECHQA_ARCHIVE_FILENAME = "TechQA.tar.gz"
BITEXT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MSDIALOG_URL = (
    "https://raw.githubusercontent.com/SCU-ChenYue/MSDialog_RL/"
    "main/test_MSDialog.jsonl"
)
TECHQA_QA_REVISION = "a906f9e27d047c5318b6ea31976327395cbd0650"
TECHQA_DOCUMENT_REVISION = "60437bc79ab217679682217598a3693cab78365b"
TECHQA_ARCHIVE_SHA256 = "6b094ef9a69718f727ce8d7e15c4d961e51032cefaa952e0d6af9d176d7ba118"
BITEXT_REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"

TECHQA_SPLIT_SIZES = {"train": 450, "validation": 160, "test": 11}
TECHQA_DOCUMENT_SAMPLE_SIZE = 5_000
BITEXT_SAMPLE_SIZE = 2_000
MSDIALOG_SAMPLE_SIZE = 500

OUTPUT_FILES = (
    Path("techqa/qa.jsonl"),
    Path("techqa/documents.jsonl"),
    Path("bitext/tickets.jsonl"),
    Path("msdialog/conversations.jsonl"),
    Path("manifest.json"),
)

DatasetLoader = Callable[..., Any]
UrlFetcher = Callable[[str], bytes]
ArchiveFetcher = Callable[..., str]
DocumentLoader = Callable[[int, ArchiveFetcher], tuple[list[dict[str, Any]], str]]


class CorpusDownloadError(RuntimeError):
    """Raised when a source corpus cannot produce the expected output."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url: str) -> bytes:
    """Download a URL and return its content."""
    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as exc:
        raise CorpusDownloadError(f"Unable to download {url}: {exc}") from exc


def _records(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in dataset]


def _validate_fields(
    records: Sequence[Mapping[str, Any]],
    required_fields: set[str],
    corpus_name: str,
) -> None:
    for index, record in enumerate(records):
        missing = required_fields.difference(record)
        if missing:
            fields = ", ".join(sorted(missing))
            raise CorpusDownloadError(
                f"{corpus_name} record {index} is missing required fields: {fields}"
            )


def load_techqa_qa(dataset_loader: DatasetLoader = load_dataset) -> list[dict]:
    """Load TechQA question/answer pairs independently from the document corpus."""
    qa_records: list[dict[str, Any]] = []

    for split, expected_size in TECHQA_SPLIT_SIZES.items():
        try:
            split_records = _records(
                dataset_loader(
                    TECHQA_QA_DATASET,
                    split=split,
                    revision=TECHQA_QA_REVISION,
                )
            )
        except Exception as exc:
            raise CorpusDownloadError(f"Unable to load TechQA split '{split}': {exc}") from exc

        if len(split_records) != expected_size:
            raise CorpusDownloadError(
                f"TechQA split '{split}' contains {len(split_records)} rows; "
                f"expected {expected_size}"
            )
        _validate_fields(
            split_records,
            {"id", "document", "question", "answer"},
            f"TechQA/{split}",
        )

        for record in split_records:
            qa_records.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "split": split,
                }
            )

    return qa_records


def _document_from_value(source_id: str, value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        text = value
        metadata: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        text_key = next(
            (
                key
                for key in ("document", "text", "content", "body", "contents")
                if isinstance(value.get(key), str)
            ),
            None,
        )
        if text_key is None:
            return None
        text = str(value[text_key])
        metadata = {
            str(key): item
            for key, item in value.items()
            if key not in {text_key, "id", "doc_id", "document_id", "uid"}
        }
        source_id = str(
            value.get("id")
            or value.get("doc_id")
            or value.get("document_id")
            or value.get("uid")
            or source_id
        )
    else:
        return None
    if not text.strip():
        return None
    record: dict[str, Any] = {
        "id": source_id,
        "split": "technotes",
        "document": text,
    }
    if metadata:
        record["source_metadata"] = metadata
    return record


def _iter_top_level_json(stream: Any) -> Iterable[tuple[str, Any]]:
    """Incrementally parse a top-level JSON array or object."""
    reader = io.TextIOWrapper(stream, encoding="utf-8")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def fill() -> None:
        nonlocal buffer, position, eof
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = reader.read(1024 * 1024)
        if chunk:
            buffer += chunk
        else:
            eof = True

    def skip_space() -> None:
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or eof:
                return
            fill()

    def parse_value() -> Any:
        nonlocal position
        while True:
            skip_space()
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise CorpusDownloadError(f"Invalid TechQA JSON: {exc}") from exc
                fill()
                continue
            position = end
            return value

    fill()
    skip_space()
    if position >= len(buffer) or buffer[position] not in "[{":
        raise CorpusDownloadError("TechQA document JSON must contain an array or object")
    opening = buffer[position]
    position += 1

    index = 0
    while True:
        skip_space()
        closing = "]" if opening == "[" else "}"
        if position < len(buffer) and buffer[position] == closing:
            return
        if opening == "[":
            key = str(index)
            value = parse_value()
            index += 1
        else:
            key = parse_value()
            if not isinstance(key, str):
                raise CorpusDownloadError("TechQA document object keys must be strings")
            skip_space()
            if position >= len(buffer):
                fill()
                skip_space()
            if position >= len(buffer) or buffer[position] != ":":
                raise CorpusDownloadError("Invalid TechQA JSON object")
            position += 1
            value = parse_value()
        yield key, value
        skip_space()
        if position >= len(buffer):
            fill()
            skip_space()
        if position < len(buffer) and buffer[position] == ",":
            position += 1
            continue
        if position < len(buffer) and buffer[position] == closing:
            return
        raise CorpusDownloadError("Invalid separator in TechQA JSON")


def _iter_archive_documents(archive_path: Path) -> Iterable[dict[str, Any]]:
    """Stream TechNote documents directly from the official TechQA archive."""
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise CorpusDownloadError(f"Unable to open TechQA archive: {exc}") from exc

    found_member = False
    with archive:
        for member in archive:
            name = member.name.casefold()
            if (
                not member.isfile()
                or not name.endswith(".json")
                or "technote" not in name
                or "q_a" in name
            ):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            found_member = True
            with stream:
                for source_id, value in _iter_top_level_json(stream):
                    document = _document_from_value(source_id, value)
                    if document is not None:
                        yield document
    if not found_member:
        raise CorpusDownloadError("TechQA archive contains no TechNote JSON file")


def load_techqa_documents(
    seed: int,
    archive_fetcher: ArchiveFetcher = hf_hub_download,
) -> tuple[list[dict[str, Any]], str]:
    """Download, verify, and deterministically sample 5,000 TechQA documents."""
    try:
        archive_path = Path(
            archive_fetcher(
                repo_id=TECHQA_DOCUMENT_REPO,
                filename=TECHQA_ARCHIVE_FILENAME,
                repo_type="dataset",
                revision=TECHQA_DOCUMENT_REVISION,
            )
        )
    except Exception as exc:
        raise CorpusDownloadError(f"Unable to download TechQA document archive: {exc}") from exc

    archive_checksum = sha256_file(archive_path)
    if archive_checksum != TECHQA_ARCHIVE_SHA256:
        raise CorpusDownloadError(
            "TechQA archive checksum mismatch: "
            f"expected {TECHQA_ARCHIVE_SHA256}, got {archive_checksum}"
        )

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    count = 0
    for count, document in enumerate(_iter_archive_documents(archive_path), start=1):
        if count <= TECHQA_DOCUMENT_SAMPLE_SIZE:
            sample.append(document)
        else:
            replacement = rng.randrange(count)
            if replacement < TECHQA_DOCUMENT_SAMPLE_SIZE:
                sample[replacement] = document
    if count < TECHQA_DOCUMENT_SAMPLE_SIZE:
        raise CorpusDownloadError(
            f"TechQA contains {count} documents; "
            f"at least {TECHQA_DOCUMENT_SAMPLE_SIZE} are required"
        )
    return sorted(sample, key=lambda record: record["id"]), archive_checksum


def load_bitext(
    seed: int,
    dataset_loader: DatasetLoader = load_dataset,
) -> list[dict[str, Any]]:
    """Load a deterministic Bitext sample."""
    try:
        records = _records(
            dataset_loader(BITEXT_DATASET, split="train", revision=BITEXT_REVISION)
        )
    except Exception as exc:
        raise CorpusDownloadError(f"Unable to load Bitext: {exc}") from exc

    _validate_fields(
        records,
        {"flags", "instruction", "category", "intent", "response"},
        "Bitext",
    )
    if len(records) < BITEXT_SAMPLE_SIZE:
        raise CorpusDownloadError(
            f"Bitext contains {len(records)} rows; at least {BITEXT_SAMPLE_SIZE} are required"
        )

    indices = random.Random(seed).sample(range(len(records)), BITEXT_SAMPLE_SIZE)
    return [records[index] for index in indices]


def load_msdialog(seed: int, url_fetcher: UrlFetcher = fetch_url) -> tuple[list[dict], str]:
    """Download MSDialog and return a deterministic conversation sample."""
    source = url_fetcher(MSDIALOG_URL)
    source_checksum = hashlib.sha256(source).hexdigest()
    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusDownloadError(
                f"MSDialog contains invalid JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise CorpusDownloadError(
                f"MSDialog line {line_number} must contain a JSON object"
            )
        records.append(record)

    if len(records) < MSDIALOG_SAMPLE_SIZE:
        raise CorpusDownloadError(
            f"MSDialog contains {len(records)} conversations; "
            f"at least {MSDIALOG_SAMPLE_SIZE} are required"
        )

    indices = random.Random(seed).sample(range(len(records)), MSDIALOG_SAMPLE_SIZE)
    return [records[index] for index in indices], source_checksum


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def _output_metadata(root: Path, relative_path: Path, count: int) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path.as_posix(),
        "records": count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _existing_outputs(data_dir: Path) -> list[Path]:
    return [relative for relative in OUTPUT_FILES if (data_dir / relative).exists()]


def _publish(staging_dir: Path, data_dir: Path, force: bool) -> None:
    for relative_path in OUTPUT_FILES:
        source = staging_dir / relative_path
        destination = data_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if force and destination.exists():
            destination.unlink()
        source.replace(destination)


def download_corpora(
    data_dir: Path,
    seed: int = 42,
    *,
    force: bool = False,
    skip_existing: bool = False,
    dataset_loader: DatasetLoader = load_dataset,
    url_fetcher: UrlFetcher = fetch_url,
    archive_fetcher: ArchiveFetcher = hf_hub_download,
    document_loader: DocumentLoader = load_techqa_documents,
) -> Path:
    """Download, validate, and export all corpus subsets."""
    existing = _existing_outputs(data_dir)
    if existing:
        if skip_existing and len(existing) == len(OUTPUT_FILES):
            return data_dir / "manifest.json"
        if not force:
            paths = ", ".join(path.as_posix() for path in existing)
            raise CorpusDownloadError(
                f"Output already exists ({paths}). Use --force to replace it"
            )

    data_dir.mkdir(parents=True, exist_ok=True)
    staging_parent = data_dir.parent
    with tempfile.TemporaryDirectory(prefix=".corpus-", dir=staging_parent) as temporary:
        staging_dir = Path(temporary)

        techqa_qa = load_techqa_qa(dataset_loader)
        techqa_documents, techqa_archive_checksum = document_loader(
            seed,
            archive_fetcher,
        )
        bitext = load_bitext(seed, dataset_loader)
        msdialog, msdialog_source_checksum = load_msdialog(seed, url_fetcher)

        write_jsonl(staging_dir / "techqa/qa.jsonl", techqa_qa)
        write_jsonl(staging_dir / "techqa/documents.jsonl", techqa_documents)
        write_jsonl(staging_dir / "bitext/tickets.jsonl", bitext)
        write_jsonl(staging_dir / "msdialog/conversations.jsonl", msdialog)

        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "sources": {
                "techqa": {
                    "qa": {
                        "dataset": TECHQA_QA_DATASET,
                        "revision": TECHQA_QA_REVISION,
                        "splits": TECHQA_SPLIT_SIZES,
                    },
                    "documents": {
                        "dataset": TECHQA_DOCUMENT_REPO,
                        "revision": TECHQA_DOCUMENT_REVISION,
                        "archive": TECHQA_ARCHIVE_FILENAME,
                        "archive_sha256": techqa_archive_checksum,
                        "sample_size": TECHQA_DOCUMENT_SAMPLE_SIZE,
                    },
                },
                "bitext": {
                    "dataset": BITEXT_DATASET,
                    "revision": BITEXT_REVISION,
                    "split": "train",
                },
                "msdialog": {
                    "url": MSDIALOG_URL,
                    "source_sha256": msdialog_source_checksum,
                },
            },
            "outputs": {
                "techqa_qa": _output_metadata(
                    staging_dir, Path("techqa/qa.jsonl"), len(techqa_qa)
                ),
                "techqa_documents": _output_metadata(
                    staging_dir,
                    Path("techqa/documents.jsonl"),
                    len(techqa_documents),
                ),
                "bitext": _output_metadata(
                    staging_dir, Path("bitext/tickets.jsonl"), len(bitext)
                ),
                "msdialog": _output_metadata(
                    staging_dir,
                    Path("msdialog/conversations.jsonl"),
                    len(msdialog),
                ),
            },
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        try:
            _publish(staging_dir, data_dir, force)
        except OSError as exc:
            raise CorpusDownloadError(f"Unable to publish corpus files: {exc}") from exc

    return data_dir / "manifest.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument(
        "--force",
        action="store_true",
        help="Replace existing corpus outputs",
    )
    behavior.add_argument(
        "--skip-existing",
        action="store_true",
        help="Return successfully when every expected output already exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus download CLI."""
    args = parse_args(argv)
    try:
        manifest_path = download_corpora(
            data_dir=args.data_dir,
            seed=args.seed,
            force=args.force,
            skip_existing=args.skip_existing,
        )
    except CorpusDownloadError as exc:
        print(f"error: {exc}")
        return 1

    print(f"Corpus download complete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
