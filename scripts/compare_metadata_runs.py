"""Compare saved metadata runs (output/metadata/*) against the reference metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction.scoring import field_matches  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "output" / "metadata"
FIELDS = ["title", "authors", "organisation", "publication_date", "source_url"]


def main() -> None:
    schema = json.loads((PROJECT_ROOT / "data" / "schema_worked_examples.json").read_text(encoding="utf-8"))
    reference = {d["document_id"]: d for d in schema["documents"]}
    runs = sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir())

    results: dict[str, dict] = {}
    for run in runs:
        docs = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in run.glob("*.json")}
        cells, strict, lenient, seconds = {}, 0, 0, 0.0
        for doc_id, extracted in docs.items():
            seconds += extracted.get("seconds") or 0
            for field in FIELDS:
                ref = reference.get(doc_id, {})
                if field not in ref:
                    continue
                s = field_matches(field, extracted.get(field), ref[field])
                n = field_matches(field, extracted.get(field), ref[field], normalise_dates=True)
                strict += s
                lenient += n
                cells[(doc_id, field)] = (s, n)
        results[run.name] = {"cells": cells, "strict": strict, "lenient": lenient, "seconds": seconds}

    keys = sorted({k for r in results.values() for k in r["cells"]})
    width = max(len(name) for name in results) + 2
    print(f"{'document / field':<38}" + "".join(f"{name:<{width}}" for name in results))
    for doc_id, field in keys:
        row = f"{doc_id.removeprefix('doc_') + ' / ' + field:<38}"
        for r in results.values():
            cell = r["cells"].get((doc_id, field))
            mark = "-" if cell is None else ("OK" if cell[0] else "ok*" if cell[1] else "miss")
            row += f"{mark:<{width}}"
        print(row)

    print("\n(ok* = matches once date formats are normalised)\n")
    total = len(keys)
    for name, r in results.items():
        print(
            f"{name}: strict {r['strict']}/{total}, date-normalised {r['lenient']}/{total}, "
            f"{r['seconds']:.1f}s model time"
        )


if __name__ == "__main__":
    main()
