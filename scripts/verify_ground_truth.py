"""Check each ground-truth statistic's value against the PDF text.

For every statistic, finds the pages where the value appears near a
keyword for its unit, and compares them with the page the ground truth cites.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}
UNIT_KEYWORDS = {
    "deaths": r"death|fatal|case|died|fatalit",
    "fatal incidents": r"fatal|incident|case|death",
    "percent": r"%|per ?cent|percent",
    "day": r"day",
    "AUD": r"\$|million|cost|compensation|gross margin",
    "kg CO2e": r"co2|kg|emission|climate",
    "years": r"year",
    "claims": r"claim",
}
WINDOW = 160


def number_patterns(value: float) -> list[str]:
    forms = []
    if float(value).is_integer():
        n = int(value)
        forms.append(str(n))
        if n >= 1000:
            forms.append(f"{n:,}")
        if n in WORDS:
            forms.append(WORDS[n])
        if n == 25:
            forms.append("quarter")
    else:
        forms.append(str(value))
    return [rf"(?<![\d.,]){re.escape(f)}(?![\d]|,\d)" for f in forms]


def keyword_regex(unit: str) -> str:
    for key, pattern in UNIT_KEYWORDS.items():
        if key.lower() in unit.lower():
            return pattern
    return r"."


def pages_with_value(page_texts: list[str], value: float, unit: str) -> list[int]:
    keyword = re.compile(keyword_regex(unit))
    patterns = [re.compile(p) for p in number_patterns(value)]
    hits = []
    for number, text in enumerate(page_texts, start=1):
        for pattern in patterns:
            if any(
                keyword.search(text[max(0, m.start() - WINDOW): m.end() + WINDOW])
                for m in pattern.finditer(text)
            ):
                hits.append(number)
                break
    return hits


def cited_pages(page) -> list[int] | None:
    if isinstance(page, int):
        return [page]
    if isinstance(page, str) and re.fullmatch(r"\d+-\d+", page):
        a, b = map(int, page.split("-"))
        return list(range(a, b + 1))
    return None


def best_offset(records: list[dict]) -> int:
    def score(offset: int) -> int:
        return sum(
            1 for r in records if r["cited"] and {c + offset for c in r["cited"]} & set(r["pages"])
        )
    return max(range(-3, 6), key=lambda o: (score(o), -abs(o)))


def main() -> None:
    truth = json.loads((DATA_DIR / "ground_truth_all_documents.json").read_text(encoding="utf-8"))
    manifest = {e["document_id"]: e["filename"] for e in
                json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))["documents"]}

    texts: dict[str, list[str]] = {}
    for doc_id, filename in manifest.items():
        document = load_document(DATA_DIR / "raw_pdfs" / filename, doc_id)
        texts[doc_id] = [re.sub(r"\s+", " ", p.text).lower() for p in document.pages]

    records: dict[str, list[dict]] = {doc_id: [] for doc_id in manifest}
    for stat in truth["statistics"]:
        record = {"stat": stat, "cited": cited_pages(stat.get("page")), "pages": []}
        if stat["value"] is not None:
            record["pages"] = pages_with_value(texts[stat["document_id"]], stat["value"], stat["unit"])
        records[stat["document_id"]].append(record)

    summary = {"value found, page matches": 0, "value found, page differs": 0,
               "value not found": 0, "no numeric value to check": 0}
    for doc_id, recs in records.items():
        offset = best_offset(recs)
        print(f"\n{doc_id}  (best page offset: cited + {offset} = PDF page)")
        for r in recs:
            stat = r["stat"]
            label = f"  {stat['id']:<16}"
            if stat["value"] is None:
                summary["no numeric value to check"] += 1
                print(f"{label} no numeric value ({stat['value_status'][:30]})")
                continue
            pages = r["pages"]
            if not pages:
                status = "value not found"
            elif r["cited"] and {c + offset for c in r["cited"]} & set(pages):
                status = "value found, page matches"
            else:
                status = "value found, page differs"
            summary[status] += 1
            print(f"{label} {status:<27} value={stat['value']!s:<7} cited={stat.get('page')!s:<9} found on {pages}")

    print()
    for status, n in summary.items():
        print(f"{status}: {n}")


if __name__ == "__main__":
    main()
