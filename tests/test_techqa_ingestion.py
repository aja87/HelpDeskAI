from __future__ import annotations

from helpdeskai.ingestion.dedup import deduplicate_documents
from helpdeskai.ingestion.enrich import enrich_documents, extract_metadata
from helpdeskai.ingestion.extract import extract_techqa_documents
from helpdeskai.ingestion.normalize import clean_document, normalize_documents, normalize_text


def test_clean_document_extracts_html_and_normalizes_unicode() -> None:
    raw = """
    <html><body><h1>IBM HTTP Server\u00a0\u2014 Guide</h1>
    <script>ignored()</script><p>First&nbsp;paragraph.</p><p>Second\u2026</p></body></html>
    """

    text, contained_html = clean_document(raw)

    assert contained_html is True
    assert "ignored" not in text
    assert "IBM HTTP Server - Guide" in text
    assert "First paragraph." in text
    assert "Second..." in text


def test_normalize_text_repairs_spacing_and_punctuation() -> None:
    assert normalize_text("  One\u00a0\u2014  two\r\n\r\n\r\nThree  ") == "One - two\n\nThree"


def test_extract_metadata_is_conservative_and_traceable() -> None:
    text = (
        "IBM WebSphere Application Server version 9.0 - United States "
        "was; server TECHNOTE (TROUBLESHOOTING)\n\n"
        "Published: January 23, 2018\n"
        "Problem details"
    )

    metadata = extract_metadata(text)

    assert metadata["title"] == "WebSphere Application Server version 9.0"
    assert metadata["country"] == "United States"
    assert metadata["keywords"] == ["was", "server"]
    assert metadata["category"]["value"] == "technote_troubleshooting"
    assert metadata["product"]["value"] == "WebSphere Application Server"
    assert metadata["versions"]["value"] == ["9.0"]
    assert metadata["date"]["value"] == "2018-01-23"


def test_extract_metadata_does_not_guess_unlabelled_values() -> None:
    metadata = extract_metadata(
        "IBM How to solve a generic issue - United States TECHNOTE (FAQ)\n"
        "The incident happened on 04/02/2015."
    )

    assert metadata["product"]["value"] is None
    assert metadata["versions"]["value"] is None
    assert metadata["date"]["value"] is None


def test_stages_deduplicate_and_preserve_aliases() -> None:
    documents = [
        {
            "id": "TRAIN_Q001",
            "split": "train",
            "document": "<p>IBM HTTP Server - United States TECHNOTE (FAQ)</p>",
        },
        {
            "id": "DEV_Q001",
            "split": "validation",
            "document": "IBM HTTP Server - United States TECHNOTE (FAQ)",
        },
        {
            "id": "TEST_Q001",
            "split": "test",
            "document": "A different document",
        },
    ]
    extracted = extract_techqa_documents(documents)
    normalized = normalize_documents(extracted)
    canonical = deduplicate_documents(normalized)
    enriched = enrich_documents(canonical)

    assert len(enriched) == 2
    duplicate = next(record for record in enriched if record["source_record_count"] == 2)
    assert duplicate["source_ids"] == ["DEV_Q001", "TRAIN_Q001"]
    assert duplicate["splits"] == ["train", "validation"]
