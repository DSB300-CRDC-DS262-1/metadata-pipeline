"""Write a markdown checklist of metadata fields where the saved model runs
disagree with the reference, with matching PDF text beside each."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction.scoring import strict_equal  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "metadata"
CHECK_PATH = PROJECT_ROOT / "output" / "verification" / "metadata_checklist.md"
FIELDS = ["title", "authors", "organisation", "publication_date", "source_url"]

EVIDENCE_PATTERNS = {
    "title": None,
    "authors": None,
    "organisation": r"universit|department of|institute|school of|agHealth|voconiq|commissioned|prepared|funded|published by|elsevier|corporation",
    "publication_date": r"received|accepted|available online|published|issued|\b(19|20)\d\d\b",
    "source_url": r"https?://|www\.|doi",
}


def evidence(field: str, pages: list[tuple[int, str]]) -> list[str]:
    lines: list[str] = []
    if field in ("title", "authors"):
        first_page, text = pages[0]
        body = [l.strip() for l in text.splitlines() if l.strip()]
        return [f"p{first_page}: {l}" for l in body[:14]]
    pattern = re.compile(EVIDENCE_PATTERNS[field], re.I)
    for number, text in pages:
        for line in text.splitlines():
            clean = re.sub(r"\s+", " ", line).strip()
            if clean and pattern.search(clean):
                lines.append(f"p{number}: {clean[:170]}")
    return lines[:10]


def main() -> None:
    schema = json.loads((DATA_DIR / "schema_worked_examples.json").read_text(encoding="utf-8"))
    reference = {d["document_id"]: d for d in schema["documents"]}
    manifest = {e["document_id"]: e["filename"] for e in
                json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))["documents"]}
    runs = {r.name: {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in r.glob("*.json")}
            for r in sorted(OUTPUT_ROOT.iterdir()) if r.is_dir()}

    out = ["# Metadata reference check", "",
           "Reference values come from an AI-generated file, so each item asks whether the "
           "reference or a model is right, judged against the PDF text shown.", ""]
    count = 0
    for doc_id, filename in manifest.items():
        document = load_document(DATA_DIR / "raw_pdfs" / filename, doc_id)
        head = [(p.page_number, p.text) for p in document.pages[:3]]
        tail = [(p.page_number, p.text) for p in document.pages[-1:]]
        for field in FIELDS:
            if field not in reference[doc_id]:
                continue
            ref = reference[doc_id][field]
            values = {name: docs.get(doc_id, {}).get(field) for name, docs in runs.items()}
            if all(strict_equal(v, ref) for v in values.values()):
                continue
            count += 1
            out += [f"## {count}. {doc_id.removeprefix('doc_')} / {field}", "",
                    f"- [ ] reference: `{ref}`"]
            out += [f"- [ ] {name}: `{value}`" for name, value in values.items()]
            out += ["", "PDF text:", "```"]
            out += evidence(field, head if field != "source_url" else head + tail) or ["(no matching text found)"]
            out += ["```", ""]
    CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECK_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {CHECK_PATH.relative_to(PROJECT_ROOT)} ({count} items)")


if __name__ == "__main__":
    main()
