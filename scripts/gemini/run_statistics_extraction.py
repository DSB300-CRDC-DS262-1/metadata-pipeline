"""Runs statistics extraction across the manifest documents and prints
the results, plus a quick recall check against
data/ground_truth_all_documents.json.

Mirrors scripts/metadata/run_metadata_extraction.py in structure. This is a
quick-look tool for development, not the real scoring script -- it
matches ground truth entries to extracted statistics by
(document_id, variable) exact string match, which will under-count
recall whenever the model phrases a variable slightly differently to
the ground truth's wording. Treat the printed recall number as a
rough signal while iterating on the prompt, not a final metric.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crdc_pipeline.extraction.gemini.statistics import extract_statistics, make_client  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "document_manifest.json"
RAW_PDFS_DIR = ROOT / "data" / "raw_pdfs"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth_all_documents.json"
OUTPUT_DIR = ROOT / "output" / "gemini" / "statistics"


def main() -> None:
    load_dotenv(ROOT / ".env")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ground_truth_by_doc: dict[str, list[dict]] = {}
    for stat in ground_truth["statistics"]:
        ground_truth_by_doc.setdefault(stat["document_id"], []).append(stat)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()

    total_found = 0
    total_expected = 0

    for entry in manifest["documents"]:
        document_id = entry["document_id"]
        filename = entry.get("filename")
        if not filename:
            print(f"[skip] {document_id}: no filename set in manifest")
            continue

        pdf_path = RAW_PDFS_DIR / filename
        if not pdf_path.is_file():
            print(f"[skip] {document_id}: {pdf_path} not found")
            continue

        print(f"\n=== {document_id} ({filename}) ===")
        document = load_document(pdf_path, document_id=document_id)
        pages = [p.text for p in document.pages]

        result = extract_statistics(pages, client)
        print(f"Extracted {len(result.statistics)} statistics (model: {result.resolved_model})")

        out_path = OUTPUT_DIR / f"{document_id}.json"
        out_path.write_text(
            json.dumps([s.model_dump() for s in result.statistics], indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")

        # Rough recall check against ground truth for this document.
        expected = ground_truth_by_doc.get(document_id, [])
        expected_variables = {e["variable"] for e in expected}
        found_variables = {s.variable for s in result.statistics}
        matched = expected_variables & found_variables

        total_found += len(matched)
        total_expected += len(expected_variables)

        print(f"Rough recall for this document: {len(matched)}/{len(expected_variables)}")
        missed = expected_variables - found_variables
        if missed:
            print("  Missed (by exact variable-name match, may be phrased differently):")
            for variable in sorted(missed):
                print(f"    - {variable}")

    if total_expected:
        print(f"\n=== Overall rough recall: {total_found}/{total_expected} "
              f"({100 * total_found / total_expected:.1f}%) ===")


if __name__ == "__main__":
    main()
