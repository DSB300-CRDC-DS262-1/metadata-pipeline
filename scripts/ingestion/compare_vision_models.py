"""Compare local vision models on page reading (Voconiq) and symbol learning (Hedayati).

Usage: python scripts/ingestion/compare_vision_models.py gemma4:e4b gemma4:12b
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction.vision import (  # noqa: E402
    make_page_transcriber,
    make_vision_backend,
    make_word_reader,
)
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402
from crdc_pipeline.ingestion.symbols import SymbolRules  # noqa: E402

PDFS = PROJECT_ROOT / "data" / "raw_pdfs"

# Prose values the Voconiq text layer damages, each confirmed by eye on the rendered page.
PAGE_CHECKS = [
    (8, r"way of life in Australia\W{0,3}\s*73%"),
    (8, r"where it operates\W{0,3}\s*\(80%\)"),
    (11, r"water responsibly at 47%"),
    (11, r"is appropriate\W{0,3}\s*at 41%"),
    (11, r"with 28% who disagreed"),
    (15, r"forgive the mistake\W{0,3}\s*\(47%\)"),
    (19, r"\(49% neutral"),
    (19, r"\(46% neutral"),
    (19, r"59% of participants agreed"),
]

# What each mislabelled Hedayati glyph really is; dashes count as equivalent.
EXPECTED_RULES = {
    "AdvPS44A44B|e": {"–", "-", "—"},
    "AdvPS44A44B|d": {"—", "–", "-"},
    "AdvP4C4E74|(cid:2)": {"−", "-", "–"},
    "AdvP4C4E74|¼": {"="},
    "AdvP4C4E74|þ": {"+"},
}


def compare(model: str) -> None:
    backend = make_vision_backend("ollama", model)
    flat = lambda text: re.sub(r"\s+", " ", text)  # noqa: E731

    start = time.monotonic()
    doc = load_document(PDFS / "voconiq2025.pdf", "v", vision=make_page_transcriber(backend))
    page_seconds = time.monotonic() - start
    read = [p for p in doc.pages if p.extraction_method == "vision"]
    found = sum(bool(re.search(rx, flat(doc.pages[n - 1].text))) for n, rx in PAGE_CHECKS)

    rules = SymbolRules()
    start = time.monotonic()
    load_document(PDFS / "hedayati2019.pdf", "h", read_text=make_word_reader(backend), rules=rules)
    word_seconds = time.monotonic() - start
    learned = {key: rules.replacement({"text": key.split("|", 1)[1], "fontname": key.split("|")[0]})
               for key in EXPECTED_RULES}
    correct = sum(learned[k] in ok for k, ok in EXPECTED_RULES.items())

    print(f"{model}: pages {len(read)}/11 read in {page_seconds:.0f}s "
          f"({page_seconds / max(len(read), 1):.1f}s per page), prose checks {found}/{len(PAGE_CHECKS)}; "
          f"symbol rules {correct}/{len(EXPECTED_RULES)} correct in {word_seconds:.0f}s "
          f"{ {k.split('|')[1]: v for k, v in learned.items()} }")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["gemma4:e4b"]:
        compare(name)
