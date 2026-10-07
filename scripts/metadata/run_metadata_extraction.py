"""Run metadata extraction on the sample documents and compare with the reference.

Usage: python scripts/run_metadata_extraction.py [--model gemma4:12b] [--backend gemini]
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

from crdc_pipeline.extraction import extract_metadata, make_backend  # noqa: E402
from crdc_pipeline.extraction.metadata.scoring import SCORED_FIELDS, field_value, matches  # noqa: E402
from crdc_pipeline.extraction.vision import make_page_transcriber, make_vision_backend  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402
from crdc_pipeline.reference import metadata_reference  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "metadata"


def load_manifest() -> list[dict]:
    manifest = json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))
    return manifest["documents"]


def load_reference_metadata() -> dict[str, dict]:
    return metadata_reference()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="ollama")
    parser.add_argument("--model", help="model name (defaults: gemma4:e4b local, gemini-3.5-flash-lite)")
    parser.add_argument("--think", choices=["default", "on", "off"], default="default",
                        help="reasoning step for local models that support it")
    parser.add_argument("--seed", type=int, help="fixed sampling seed for local models")
    parser.add_argument("--num-ctx", type=int, help="context window for local models (default 32768)")
    parser.add_argument("--num-gpu", type=int, help="model layers to put on the GPU (for smaller GPUs)")
    parser.add_argument("--vision", action="store_true",
                        help="read pages with garbled or no text from the page image (local vision model)")
    parser.add_argument("--suffix", default="", help="appended to the output folder, e.g. r2 for a repeat run")
    parser.add_argument("--documents", nargs="*", help="document_ids to run (default all)")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    think = {"default": None, "on": True, "off": False}[args.think]
    options = {"think": think, "seed": args.seed} if args.backend == "ollama" else {}
    if args.num_ctx and args.backend == "ollama":
        options["num_ctx"] = args.num_ctx
    if args.num_gpu is not None and args.backend == "ollama":
        options["num_gpu"] = args.num_gpu
    backend = make_backend(args.backend, args.model, **options)
    transcribe = make_page_transcriber(make_vision_backend("ollama")) if args.vision else None
    reference = load_reference_metadata()
    out_dir = OUTPUT_ROOT / slug(backend.label + (f"__{args.suffix}" if args.suffix else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    matched = compared = 0
    total_seconds = 0.0

    for entry in load_manifest():
        document_id = entry["document_id"]
        if args.documents and document_id not in args.documents:
            continue
        filename = entry.get("filename")
        pdf_path = DATA_DIR / "raw_pdfs" / filename if filename else None
        if not pdf_path or not pdf_path.is_file():
            print(f"[skip] {document_id}: PDF not found")
            continue

        print(f"\n=== {document_id} ({filename}) ===")
        document = load_document(pdf_path, document_id=document_id, vision=transcribe, vision_on_unreadable=False)
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
            "retrieved_from": entry.get("retrieved_from"),
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
        found = metadata.model_dump()
        for field in SCORED_FIELDS:
            extracted, expected = field_value(found, field), field_value(ref, field)
            ok = matches(field, extracted, expected)
            matched += ok
            compared += 1
            print(f"  {'OK' if ok else '??'} {field.removeprefix('journal_article.'):<17} {extracted!s:<45} ref: {expected!s}")

    print(
        f"\n{backend.label}: {matched}/{compared} reference fields matched, "
        f"{total_seconds:.1f}s total model time"
    )


if __name__ == "__main__":
    main()
