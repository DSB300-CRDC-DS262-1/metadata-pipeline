"""Shared fixtures: the sample PDFs' text, for checking reference quotes against their page."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "data" / "document_manifest.json").read_text(encoding="utf-8"))
QUOTE_MARKS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text.translate(QUOTE_MARKS))


@pytest.fixture(scope="session")
def quote_checker():
    """A function (document_id, records) -> records whose quote is not on their page.

    Each page is read as the loader reads it and also column by column, for pages where the loader
    mixes two columns line by line. Skips the test when the PDF is not present.
    """
    cache: dict[str, tuple[list[str], list[str]]] = {}

    def texts(document_id: str) -> tuple[list[str], list[str]]:
        if document_id not in cache:
            entry = next(d for d in MANIFEST["documents"] if d["document_id"] == document_id)
            pdf = ROOT / "data" / "raw_pdfs" / entry["filename"]
            if not pdf.is_file():
                pytest.skip(f"{pdf.name} not present")
            import pdfplumber

            from crdc_pipeline.ingestion.loader import DEFAULT_RULES_PATH, _normalise, load_document
            from crdc_pipeline.ingestion.symbols import SymbolRules

            loaded = [p.text for p in load_document(pdf, document_id).pages]
            rules = SymbolRules(DEFAULT_RULES_PATH)
            columns = []
            with pdfplumber.open(pdf) as source:
                for page in source.pages:
                    rules.apply(page.chars)
                    left, top, right, bottom = page.bbox  # the page box need not start at 0
                    middle = (left + right) / 2
                    halves = [(left, top, middle, bottom), (middle, top, right, bottom)]
                    columns.append("\n".join(_normalise(page.crop(box).extract_text() or "") for box in halves))
            cache[document_id] = (loaded, columns)
        return cache[document_id]

    def missing(document_id: str, records: list[dict]) -> list[str]:
        loaded, columns = texts(document_id)
        not_found = []
        for record in records:
            page_texts = [loaded[record["page"] - 1], columns[record["page"] - 1]]
            # Words split over a line break ("in-\nventory") are checked both joined and as printed.
            candidates = [squash(t) for t in page_texts] + [squash(re.sub(r"-\s*\n\s*", "", t)) for t in page_texts]
            quote = squash(record["quote"])
            if record.get("read_from_image"):
                # Digits the text layer lost were read from the page image; the wording around them must still match.
                candidates = [re.sub(r"[\d%.]", "", c) for c in candidates]
                quote = re.sub(r"[\d%.]", "", quote)
            if not any(quote in c for c in candidates):
                not_found.append(f"{record.get('statistic_id') or record['item_id']} (page {record['page']})")
        return not_found

    return missing
