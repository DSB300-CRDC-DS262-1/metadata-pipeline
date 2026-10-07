"""Re-read pages whose PDF text layer has unreadable characters, using a vision model on the page image.

Usage: python scripts/transcribe_unreadable_pages.py --backend ollama --model gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction.vision import make_page_transcriber, make_vision_backend  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "vision"
PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
WORD = re.compile(r"[a-z]{4,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="ollama")
    parser.add_argument("--model", help="defaults to gemma4:e4b for local models")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    transcribe = make_page_transcriber(make_vision_backend(args.backend, args.model))

    manifest = json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest["documents"]:
        path = DATA_DIR / "raw_pdfs" / entry["filename"]
        text_layer = load_document(path, entry["document_id"])
        flagged = [p for p in text_layer.pages if p.unreadable_glyphs]
        if not flagged:
            continue
        vision = load_document(path, entry["document_id"], vision=transcribe)
        print(f"\n=== {entry['document_id']}: {len(flagged)} pages with unreadable characters ===")
        for page in flagged:
            seen = vision.pages[page.page_number - 1].text
            out = OUT_DIR / f"{entry['document_id']}_p{page.page_number}.txt"
            out.write_text(seen, encoding="utf-8")
            print(f"  p{page.page_number}: {page.unreadable_glyphs} unreadable characters, "
                  f"percentages found: text layer {len(PERCENT.findall(page.text))}, "
                  f"image {len(PERCENT.findall(seen))}, "
                  f"text-layer words found in image text {_overlap(page.text, seen):.0%}")


def _overlap(text_layer: str, seen: str) -> float:
    """Share of the text layer's words that the image transcription also contains."""
    words = WORD.findall(text_layer.lower())
    found = set(WORD.findall(seen.lower()))
    return sum(w in found for w in words) / len(words) if words else 0.0


if __name__ == "__main__":
    main()
