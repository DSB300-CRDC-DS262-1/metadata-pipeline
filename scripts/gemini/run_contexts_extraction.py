"""Runs context extraction across the manifest documents and prints the
   results, plus a quick recall check against
   data/ground_truth_all_documents.json.

Mirrors scripts/metadata/run_metadata_extraction.py in structure. This is a
quick-look tool for development, not the real scoring script.

The recall calculation matches ground-truth context entries to
extracted contexts using a simple exact-string comparison of
(document_id, context_type, value), which will under-count recall
whenever the model normalises wording, chooses a different but
equivalent value representation, or extracts a broader/narrower
context than the ground truth.

Examples of expected mismatches include:
   - "Northwest NSW" vs "Northwest New South Wales"
   - "National Coroners Information System" vs "NCIS"
   - Normalised date ranges vs raw text periods
   - Slightly different summaries of assumptions or limitations

Consequently, the printed recall figure should be treated as a rough
signal when iterating on prompts and schemas rather than a final
evaluation metric.

This script evaluates only whether expected contextual information was
identified. It does not evaluate:
   - context scope classification (global, section or finding-specific)
   - context-to-statistic linking
   - context deduplication across documents
   - corpus-level context reconciliation or canonicalisation
   - provenance quality beyond the extracted context records themselves
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crdc_pipeline.extraction.gemini.contexts import extract_contexts, make_client, assign_context_ids  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "document_manifest.json"
RAW_PDFS_DIR = ROOT / "data" / "raw_pdfs"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth_all_documents.json"
OUTPUT_DIR = ROOT / "output" / "gemini" / "contexts"


def main() -> None:
    load_dotenv(ROOT / ".env")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ground_truth_by_doc: dict[str, list[dict]] = {}
    for doc_contexts in ground_truth["contexts"]:
        document_id = doc_contexts["document"]
        ground_truth_by_doc.setdefault(document_id, []).extend(doc_contexts["contexts"])

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

        result = extract_contexts(document_id, pages, client)
#         result = {}
#         result["contexts"] = assign_context_ids(init_result.contexts, document_id)
        print(f"Extracted {len(result.contexts)} contexts (model: {result.resolved_model})")

        out_path = OUTPUT_DIR / f"{document_id}.json"
        out_path.write_text(
            json.dumps([s.model_dump() for s in result.contexts], indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")

        # recall check against ground truth for this document.
        expected = ground_truth_by_doc.get(document_id, [])
        expected_contexts = {
            (
                c["context_type"],
                json.dumps(c["value"], sort_keys=True)
                if isinstance(c["value"], dict)
                else str(c["value"])
            )
            for c in expected
        }
        found_contexts = {
            (
                c.context_type,
                json.dumps(c.value.model_dump(), sort_keys=True)
                if c.value.date_range is not None
                else str(c.value.text_value)
            )
            for c in result.contexts
        }
        matched = expected_contexts & found_contexts

        total_found += len(matched)
        total_expected += len(expected_contexts)

        print(f"Rough recall for this document: {len(matched)}/{len(expected_contexts)}")
        missed = expected_contexts - found_contexts
        if missed:
            print("  Missed (by exact context-value match, may be phrased differently):")
            for ctx in sorted(missed):
                print(f"    - {ctx}")

    if total_expected:
        print(f"\n=== Overall rough recall: {total_found}/{total_expected} "
              f"({100 * total_found / total_expected:.1f}%) ===")


if __name__ == "__main__":
    main()
