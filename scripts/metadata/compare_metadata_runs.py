"""Compare saved metadata runs (output/metadata/*) against the reference, grouping repeat runs.

Repeat runs are folders named <model>__r2, <model>__r3 and so on.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction.metadata.scoring import SCORED_FIELDS, field_value, matches  # noqa: E402
from crdc_pipeline.reference import metadata_reference  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "output" / "metadata"


def spread(values: list[float]) -> str:
    mean = sum(values) / len(values)
    return f"{mean:.1f}" + (f" ({min(values):g}-{max(values):g})" if len(set(values)) > 1 else "")


def main() -> None:
    reference = metadata_reference()
    checks = [(doc, f) for doc in reference for f in SCORED_FIELDS]

    models: dict[str, list[dict]] = defaultdict(list)
    for run in sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir()):
        docs = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in run.glob("*.json")}
        if len(docs) == len(reference):
            models[run.name.split("__")[0]].append(docs)

    misses: dict[str, dict[str, int]] = {}
    unstable_by_model: dict[str, list[tuple[str, str]]] = {}
    print(f"{len(checks)} checks per run\n")
    print(f"{'model':<38}{'runs':<6}{'score':<18}{'majority':<10}{'s/doc':<8}{'changed answers'}")
    for model, runs in models.items():
        scores, seconds, field_misses = [], [], defaultdict(int)
        for docs in runs:
            ok = [matches(f, field_value(docs[d], f), field_value(reference[d], f)) for d, f in checks]
            scores.append(sum(ok))
            seconds.append(sum(doc.get("seconds") or 0 for doc in docs.values()) / len(docs))
            for (d, f), hit in zip(checks, ok):
                field_misses[f] += not hit
        unstable = [
            (d, f) for d, f in checks
            if len({json.dumps(field_value(docs[d], f), sort_keys=True) for docs in runs}) > 1
        ]
        majority = sum(matches(f, _most_common(runs, d, f), field_value(reference[d], f)) for d, f in checks)
        misses[model] = field_misses
        unstable_by_model[model] = unstable
        print(f"{model:<38}{len(runs):<6}{spread(scores):<18}{majority:<10}"
              f"{sum(seconds) / len(seconds):<8.1f}{len(unstable)}")

    print("\nAnswers that change between runs:")
    for model, unstable in unstable_by_model.items():
        if unstable:
            print(f"  {model}: " + ", ".join(f"{d.removeprefix('doc_')}/{f}" for d, f in unstable))

    print("\nMisses per field, summed over runs:")
    fields = [f for f in SCORED_FIELDS if any(m[f] for m in misses.values())]
    print(f"{'field':<20}" + "".join(f"{m[:14]:<15}" for m in misses))
    for field in fields:
        print(f"{field:<20}" + "".join(f"{misses[m][field]:<15}" for m in misses))


def _most_common(runs: list[dict], doc: str, field: str):
    """The answer most runs gave for one field, i.e. a majority vote across repeat runs."""
    votes = Counter(json.dumps(field_value(docs[doc], field), sort_keys=True) for docs in runs)
    return json.loads(votes.most_common(1)[0][0])


if __name__ == "__main__":
    main()
