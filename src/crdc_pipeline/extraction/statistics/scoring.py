"""Scoring extracted statistics against data/reference/statistics.json."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..contexts.scoring import context_links, key_conditions


FIELDS = ("value_status", "record_type", "statistic_type", "response_category", "confirmed", "period")


RELATIVE_TOLERANCE = 0.005


def close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(RELATIVE_TOLERANCE * abs(a), 1e-9)


def numbers(statistic: dict) -> list[float]:
    return [statistic[k] for k in ("value", "value_min", "value_max") if statistic.get(k) is not None]


def values_match(reference: dict, extracted: dict) -> bool:
    """Every number in the reference appears among the extracted numbers, within rounding."""
    wanted, found = numbers(reference), numbers(extracted)
    return bool(wanted) and all(any(close(w, f) for f in found) for w in wanted)


def match_statistics(reference: list[dict], extracted: list[dict]) -> list[tuple[dict, dict]]:
    """Pairs on the same page with matching values, preferring the same answer category and wording."""
    used: set[int] = set()
    pairs = []
    for ref in reference:
        candidates = [
            (i, ext) for i, ext in enumerate(extracted)
            if i not in used and ext.get("page") == ref["page"] and values_match(ref, ext)
        ]
        if not candidates:
            continue
        i, ext = max(candidates, key=lambda c: (
            (c[1].get("response_category") or "").lower() == (ref.get("response_category") or "").lower(),
            SequenceMatcher(None, ref["variable"].lower(), c[1].get("variable", "").lower()).ratio(),
        ))
        used.add(i)
        pairs.append((ref, ext))
    return pairs


def field_agrees(field: str, reference: dict, extracted: dict) -> bool:
    if field == "period":
        ref, ext = reference.get("period") or {}, extracted.get("period") or {}
        return (ref.get("start"), ref.get("end")) == (ext.get("start"), ext.get("end"))
    if field == "confirmed":
        return reference.get("confirmed", True) == extracted.get("confirmed", True)
    ref, ext = reference.get(field), extracted.get(field)
    return (ref or "").lower() == (ext or "").lower() if isinstance(ref, str) or isinstance(ext, str) else ref == ext


def score_document(reference: dict, extracted: dict) -> dict:
    """Recall, precision, field agreement and context links for one document."""
    pairs = match_statistics(reference["statistics"], extracted["statistics"])
    agreement = {}
    for field in FIELDS:
        relevant = [(r, e) for r, e in pairs if field != "response_category" or r.get("response_category")]
        agreement[field] = (sum(field_agrees(field, r, e) for r, e in relevant), len(relevant))
    matched_ids = {id(r) for r, _ in pairs}
    return {
        "reference": len(reference["statistics"]),
        "extracted": len(extracted["statistics"]),
        "matched": len(pairs),
        "agreement": agreement,
        "context_links": context_links(pairs, reference, extracted),
        "key_conditions": key_conditions(pairs, reference, extracted),
        "missed": [r for r in reference["statistics"] if id(r) not in matched_ids],
        "unmatched": [e for e in extracted["statistics"] if all(e is not x for _, x in pairs)],
    }
