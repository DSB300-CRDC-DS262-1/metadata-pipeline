"""Run statistics and context extraction on the sample documents and score it against the reference.

Usage: python scripts/run_statistics_extraction.py [--model gemma4:e4b] [--documents doc_aghealth2019]
       python scripts/run_statistics_extraction.py --score-only output/statistics/<run>
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

from crdc_pipeline.extraction import make_backend  # noqa: E402
from crdc_pipeline.extraction.statistics.scoring import FIELDS, score_document  # noqa: E402
from crdc_pipeline.extraction.contexts.schema import ContextItem  # noqa: E402
from crdc_pipeline.extraction.statistics.schema import Statistic  # noqa: E402
from crdc_pipeline.extraction.contexts.links import relink  # noqa: E402
from crdc_pipeline.extraction.statistics.extract import extract_findings  # noqa: E402
from crdc_pipeline.extraction.vision import make_page_transcriber, make_vision_backend  # noqa: E402
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402
from crdc_pipeline.reference import findings_reference  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "statistics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="ollama")
    parser.add_argument("--model", help="model name (default gemma4:e4b locally)")
    parser.add_argument("--think", choices=["default", "on", "off"], default="default")
    parser.add_argument("--num-ctx", type=int, help="context window for local models (default 32768)")
    parser.add_argument("--num-gpu", type=int, help="model layers to put on the GPU (for smaller GPUs)")
    parser.add_argument("--documents", nargs="*", help="document_ids to run (default: those in the findings reference)")
    parser.add_argument("--vision", action="store_true",
                        help="read pages with unreadable characters or garbled text from the page image (local vision model)")
    parser.add_argument("--exclude-tables", action="store_true", help="drop table and chart lines before extraction")
    parser.add_argument("--no-link", action="store_true", help="skip the separate linking pass")
    parser.add_argument("--context-backend", choices=["gemini", "ollama"], help="backend for the context steps (default: --backend)")
    parser.add_argument("--context-model", help="model for the context, scope and linking steps (default: --model)")
    parser.add_argument("--context-think", choices=["default", "on", "off"], default="default")
    parser.add_argument("--whole-document", action="store_true", help="context and linking steps over the whole document in one call")
    parser.add_argument("--no-scope", action="store_true", help="skip linking whole-study items to every statistic")
    parser.add_argument("--no-dedupe", action="store_true", help="keep numbers restated elsewhere in the document")
    parser.add_argument("--suffix", default="", help="appended to the output folder, e.g. r2")
    parser.add_argument("--relink-from", type=Path, help="redo only the linking of a saved run, in one whole-document call per document, using --context-backend/--context-model")
    parser.add_argument("--score-only", type=Path, help="score a saved run folder instead of extracting")
    return parser.parse_args()


def link_scores(links) -> str:
    correct, wanted, given = links
    return f"recall {correct / max(wanted, 1):.0%} precision {correct / max(given, 1):.0%}"


def report(scores: dict[str, dict]) -> None:
    total = {"reference": 0, "extracted": 0, "matched": 0}
    agreement = {f: [0, 0] for f in FIELDS}
    context = [0, 0, 0]
    for document_id, s in scores.items():
        for k in total:
            total[k] += s[k]
        for f in FIELDS:
            agreement[f][0] += s["agreement"][f][0]
            agreement[f][1] += s["agreement"][f][1]
        for k in range(3):
            context[k] += s["context_links"][k]
        print(f"{document_id:18} found {s['matched']:3}/{s['reference']:<3} reference statistics, "
              f"{s['extracted'] - s['matched']:3} extra, context links {link_scores(s['context_links'])}")
    recall = total["matched"] / max(total["reference"], 1)
    precision = total["matched"] / max(total["extracted"], 1)
    print(f"\nrecall {recall:.0%} ({total['matched']}/{total['reference']}), "
          f"precision {precision:.0%} ({total['matched']}/{total['extracted']})")
    print("agreement on matched statistics: " + ", ".join(
        f"{f} {a}/{n}" for f, (a, n) in agreement.items()))
    print(f"context links: {link_scores(context)} (by specific fact, on matched statistics)")
    kinds = {}
    for s in scores.values():
        for kind, counts in s["key_conditions"].items():
            kinds.setdefault(kind, [0, 0, 0])
            for k in range(3):
                kinds[kind][k] += counts[k]
    print("key conditions (right / wrong / missing): " + ", ".join(
        f"{kind} {r}/{w}/{m}" for kind, (r, w, m) in kinds.items() if r + w + m))


def relink_run(args: argparse.Namespace, reference: dict[str, dict]) -> None:
    """Keeps a saved run's context items and statistics and redoes only their links."""
    kind = args.context_backend or args.backend
    backend = make_backend(kind, args.context_model)
    folder = args.relink_from.parent / f"{args.relink_from.name}__relinked-{re.sub(r'[^A-Za-z0-9._-]+', '-', backend.label)}"
    folder.mkdir(parents=True, exist_ok=True)
    scores = {}
    for path in sorted(args.relink_from.glob("doc_*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        document_id = saved["document_id"]
        if args.documents and document_id not in args.documents:
            continue
        items = [ContextItem(**i) for i in saved["context_items"]]
        statistics = [Statistic(**s) for s in saved["statistics"]]
        linked, contexts, assigned, failed = relink(items, statistics, backend, document_id)
        saved.update(statistics=[json.loads(s.model_dump_json()) for s in linked],
                     contexts=[c.model_dump() for c in contexts], context_ids=assigned,
                     failed_steps=saved.get("failed_steps", []) + failed, study_item_ids=[])
        for statistic in saved["statistics"]:
            statistic["context_id"] = assigned.get(statistic["statistic_id"])
        (folder / path.name).write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{document_id}: {len(contexts)} contexts" + "".join(f"\n  [failed] {f}" for f in failed))
        scores[document_id] = score_document(reference[document_id], saved)
    print(f"\nSaved to {folder}\n")
    report(scores)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    reference = {d["document_id"]: d for d in findings_reference()}

    if args.score_only:
        scores = {}
        for path in sorted(args.score_only.glob("doc_*.json")):
            saved = json.loads(path.read_text(encoding="utf-8"))
            scores[saved["document_id"]] = score_document(reference[saved["document_id"]], saved)
        report(scores)
        return

    load_dotenv(PROJECT_ROOT / ".env")
    if args.relink_from:
        relink_run(args, reference)
        return
    think = {"default": None, "on": True, "off": False}[args.think]
    options = {}
    if args.backend == "ollama":
        options = {k: v for k, v in {"think": think, "num_ctx": args.num_ctx, "num_gpu": args.num_gpu}.items()
                   if v is not None}
    backend = make_backend(args.backend, args.model, **options)
    context_backend = None
    label = backend.label
    if args.context_model or args.context_backend:
        context_kind = args.context_backend or args.backend
        context_options = {}
        if context_kind == "ollama":
            context_options = {**options, "think": {"default": None, "on": True, "off": False}[args.context_think]}
            context_options = {k: v for k, v in context_options.items() if v is not None}
        context_backend = make_backend(context_kind, args.context_model, **context_options)
        label += "__context-" + context_backend.label
    folder = OUTPUT_ROOT / (re.sub(r"[^A-Za-z0-9._-]+", "-", label) + ("__notables" if args.exclude_tables else "") + (f"__{args.suffix}" if args.suffix else ""))
    folder.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))["documents"]
    scores = {}
    wanted = args.documents or list(reference)
    transcribe = make_page_transcriber(make_vision_backend("ollama")) if args.vision else None
    for entry in manifest:
        document_id = entry["document_id"]
        if document_id not in wanted:
            continue
        pdf = DATA_DIR / "raw_pdfs" / entry["filename"]
        if not pdf.is_file():
            print(f"[skip] {document_id}: {pdf.name} not found")
            continue
        pages = [p.text for p in load_document(pdf, document_id, vision=transcribe).pages]
        result = extract_findings(pages, backend, document_id, exclude_tables=args.exclude_tables, link=not args.no_link, scope=not args.no_scope,
                                  context_backend=context_backend, whole_document=args.whole_document,
                                  dedupe=not args.no_dedupe)
        saved = {"document_id": document_id, **json.loads(result.model_dump_json())}
        for statistic in saved["statistics"]:
            statistic["context_id"] = saved["context_ids"].get(statistic["statistic_id"])
        (folder / f"{document_id}.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{document_id}: {len(result.statistics)} statistics, {len(result.context_items)} context items, "
              f"{len(result.contexts)} contexts in {result.seconds:.0f}s")
        for step in result.failed_steps:
            print(f"  [failed] {step}")
        if document_id in reference:
            scores[document_id] = score_document(reference[document_id], saved)
    print(f"\nSaved to {folder}\n")
    report(scores)


if __name__ == "__main__":
    main()
