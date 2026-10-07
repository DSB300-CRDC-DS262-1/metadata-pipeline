"""Checks the hand-built statistics and contexts references against the schema and, when the PDFs are present, the source text."""

import json
import re
from pathlib import Path

import pytest

from crdc_pipeline.extraction.contexts.schema import ContextItem, group_contexts
from crdc_pipeline.extraction.statistics.schema import Statistic
from crdc_pipeline.reference import findings_reference

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "data" / "document_manifest.json").read_text(encoding="utf-8"))
DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def documents():
    return [pytest.param(d, id=d["document_id"]) for d in findings_reference()]


QUOTE_MARKS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text.translate(QUOTE_MARKS))


@pytest.mark.parametrize("document", documents())
def test_records_fit_the_schema_and_rules(document) -> None:
    items = [ContextItem(**i) for i in document["context_items"]]
    statistics = [Statistic(**s) for s in document["statistics"]]
    item_ids = [i.item_id for i in items]
    assert len(item_ids) == len(set(item_ids))
    assert len({s.statistic_id for s in statistics}) == len(statistics)
    for statistic in statistics:
        assert set(statistic.context_item_ids) <= set(item_ids), statistic.statistic_id
        bounded = statistic.value_status in ("lower_bound", "upper_bound", "range")
        assert (statistic.value is None) == bounded, statistic.statistic_id
        if statistic.value_status in ("lower_bound", "range"):
            assert statistic.value_min is not None, statistic.statistic_id
        if statistic.value_status in ("upper_bound", "range"):
            assert statistic.value_max is not None, statistic.statistic_id
    for period in [s.period for s in statistics] + [i.period for i in items]:
        for date in [period.start, period.end] if period else []:
            assert date is None or DATE.match(date), date


@pytest.mark.parametrize("document", documents())
def test_stored_contexts_match_the_grouping(document) -> None:
    items = [ContextItem(**i) for i in document["context_items"]]
    statistics = [Statistic(**s) for s in document["statistics"]]
    contexts, assigned = group_contexts(document["document_id"], items, statistics)
    assert document["contexts"] == [c.model_dump() for c in contexts]
    assert {s["statistic_id"]: s["context_id"] for s in document["statistics"]} == assigned


@pytest.mark.parametrize("document", documents())
def test_quotes_appear_on_their_page(document) -> None:
    entry = next(d for d in MANIFEST["documents"] if d["document_id"] == document["document_id"])
    pdf = ROOT / "data" / "raw_pdfs" / entry["filename"]
    if not pdf.is_file():
        pytest.skip(f"{pdf.name} not present")
    import pdfplumber

    from crdc_pipeline.ingestion.loader import DEFAULT_RULES_PATH, _normalise, load_document
    from crdc_pipeline.ingestion.symbols import SymbolRules

    loaded = [p.text for p in load_document(pdf, document["document_id"]).pages]
    rules = SymbolRules(DEFAULT_RULES_PATH)
    columns = []
    with pdfplumber.open(pdf) as source:
        # Each column read on its own, for pages where the loader mixes two columns line by line.
        for page in source.pages:
            rules.apply(page.chars)
            left, top, right, bottom = page.bbox  # the page box need not start at 0
            middle = (left + right) / 2
            halves = [(left, top, middle, bottom), (middle, top, right, bottom)]
            columns.append("\n".join(_normalise(page.crop(box).extract_text() or "") for box in halves))
    missing = []
    alternatives = [{**q, "item_id": i["item_id"]} for i in document["context_items"] for q in i.get("also_stated", [])]
    for record in document["context_items"] + document["statistics"] + alternatives:
        texts = [loaded[record["page"] - 1], columns[record["page"] - 1]]
        # Words split over a line break ("in-\nventory") are checked both joined and as printed.
        candidates = [squash(t) for t in texts] + [squash(re.sub(r"-\s*\n\s*", "", t)) for t in texts]
        quote = squash(record["quote"])
        if record.get("read_from_image"):
            # Digits the text layer lost were read from the page image; the wording around them must still match.
            candidates = [re.sub(r"[\d%.]", "", c) for c in candidates]
            quote = re.sub(r"[\d%.]", "", quote)
        if not any(quote in c for c in candidates):
            missing.append(f"{record.get('statistic_id') or record['item_id']} (page {record['page']})")
    assert not missing, "quotes not found: " + ", ".join(missing)
