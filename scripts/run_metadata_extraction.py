"""Run metadata extraction on the sample documents and compare with the reference.

Usage: python scripts/run_metadata_extraction.py [--backend ollama --model gemma3:12b]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction import extract_metadata, make_backend  # noqa: E402
from crdc_pipeline.extraction.scoring import strict_equal  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "metadata"

COMPARED_FIELDS = ["title", "authors", "organisation", "publisher", "funder", "publication_date", "submission_date", "acceptance_date", "source_url"]


def load_manifest() -> list[dict]:
    manifest = json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))
    return manifest["documents"]


def load_reference_metadata() -> dict[str, dict]:
    schema = json.loads((DATA_DIR / "schema_worked_examples.json").read_text(encoding="utf-8"))
    return {d["document_id"]: d for d in schema["documents"]}


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="gemini")
    parser.add_argument("--model", help="model name (required for ollama)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    backend = make_backend(args.backend, args.model)
    reference = load_reference_metadata()
    out_dir = OUTPUT_ROOT / slug(backend.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched = compared = 0
    total_seconds = 0.0

    for entry in load_manifest():
        document_id = entry["document_id"]
        filename = entry.get("filename")
        pdf_path = DATA_DIR / "raw_pdfs" / filename if filename else None
        if not pdf_path or not pdf_path.is_file():
            print(f"[skip] {document_id}: PDF not found")
            continue

        print(f"\n=== {document_id} ({filename}) ===")
        document = load_document(pdf_path, document_id=document_id)
        extraction = extract_metadata([p.text for p in document.pages], backend)
        metadata = extraction.metadata
        total_seconds += extraction.seconds

        result = {
            "document_id": document_id,
            "page_count": document.page_count,
            "resolved_model": extraction.resolved_model,
            "seconds": round(extraction.seconds, 2),
            "prompt_tokens": extraction.prompt_tokens,
            "output_tokens": extraction.output_tokens,
            **metadata.model_dump(),
        }
        out_path = out_dir / f"{document_id}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"  wrote {out_path.relative_to(PROJECT_ROOT)} "
            f"({extraction.seconds:.1f}s, {extraction.prompt_tokens} in / "
            f"{extraction.output_tokens} out tokens)"
        )

        ref = reference.get(document_id, {})
        for field in COMPARED_FIELDS:
            extracted, expected = getattr(metadata, field), ref.get(field)
            if expected is None and field not in ref:
                continue
            ok = strict_equal(extracted, expected)
            matched += ok
            compared += 1
            print(f"  {'OK' if ok else '??'} {field:<17} {extracted!s:<45} ref: {expected!s}")

    print(
        f"\n{backend.label}: {matched}/{compared} reference fields matched, "
        f"{total_seconds:.1f}s total model time"
    )


if __name__ == "__main__":
    main()
