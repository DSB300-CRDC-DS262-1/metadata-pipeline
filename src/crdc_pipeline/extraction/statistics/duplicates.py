"""The same number listed more than once: restatements, conversions in brackets and ranges."""

from __future__ import annotations

import re

from .prompts import DUPLICATE_PROMPT, MAX_DUPLICATE_PAIRS
from .schema import Statistic


def _numbers(statistic: Statistic) -> tuple:
    return (statistic.value, statistic.value_min, statistic.value_max)


def _units_agree(first: Statistic, second: Statistic) -> bool:
    """Units that agree, in any word order ("AUD million", "million AUD")."""
    a, b = (first.unit or "").strip().lower(), (second.unit or "").strip().lower()
    return not a or not b or a == b or a in b or b in a or set(a.split()) == set(b.split())


def _decimals(number: float) -> int:
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def _same_numbers(first: Statistic, second: Statistic) -> bool:
    """The same printed numbers; or, with units given on both sides that agree, the same number stated as
    a value and as a bound ("approaching $3.5 million", "approximately $3.5 million") or rounded (10.28
    restated as 10.3)."""
    if _numbers(first) == _numbers(second):
        return True
    if not ((first.unit or "").strip() and (second.unit or "").strip() and _units_agree(first, second)):
        return False
    a = [n for n in _numbers(first) if n is not None]
    b = [n for n in _numbers(second) if n is not None]
    if len(a) != 1 or len(b) != 1:
        return False
    x, y = a[0], b[0]
    places = min(_decimals(x), _decimals(y))
    return x == y or (places >= 1 and abs(x - y) <= 0.5 * 10 ** -places + 1e-9)


def duplicate_candidates(statistics: list[Statistic]) -> list[tuple[Statistic, Statistic]]:
    """Pairs with the same numbers and compatible units, each later number paired with its earliest match."""
    pairs = []
    for j, later in enumerate(statistics):
        if all(n is None for n in _numbers(later)):
            continue
        earlier = next((s for s in statistics[:j] if _same_numbers(s, later) and _units_agree(s, later)), None)
        if earlier is not None:
            pairs.append((earlier, later))
    return pairs[:MAX_DUPLICATE_PAIRS]


def duplicate_prompt(pairs: list[tuple[Statistic, Statistic]]) -> str:
    def line(s: Statistic) -> str:
        return f"p{s.page} | {s.variable} | {s.reported_text} | {' '.join(s.quote.split())}"

    listed = "\n".join(f"pair_{k:03d}\n  A: {line(a)}\n  B: {line(b)}" for k, (a, b) in enumerate(pairs, start=1))
    return DUPLICATE_PROMPT.format(pairs=listed)


def pair_numbers(pair_ids: list[str], count: int) -> set[int]:
    """'pair_007' -> 7, ignoring IDs the model made up."""
    found = (re.search(r"(\d+)", i) for i in pair_ids)
    return {int(m.group(1)) for m in found if m and 1 <= int(m.group(1)) <= count}


RANGING = re.compile(
    r"\b(?:rang\w*\s+(?:from\s+)?(?P<low>[^\s,;]+)(?P<middle>[^.;]{0,80}?)\bto|between\s+(?P<low2>[^\s,;]+)\s+and)\s+"
    r"(?P<high>[^\s,;]+)", re.I)


def merge_ranges(statistics: list[Statistic]) -> list[Statistic]:
    """Two numbers a sentence joins as a range ("ranging from 0.0033 for ... to 0.0105 for ...") are one
    statistic with a lowest and highest value, as printed."""
    merged, dropped = [], set()
    for low in statistics:
        if id(low) in dropped:
            continue
        found = RANGING.search(" ".join(low.quote.split()))
        high = None
        first = (found.group("low") or found.group("low2")) if found else None
        if found and low.reported_text.strip() == first and low.value is not None:
            high = next((s for s in statistics if s is not low and s.page == low.page and id(s) not in dropped
                         and s.reported_text.strip() == found.group("high") and s.value is not None), None)
        if high is None:
            merged.append(low)
            continue
        dropped.add(id(high))
        merged.append(low.model_copy(update={
            "reported_text": f"{first} ... to {found.group('high')}",
            "value": None, "value_min": low.value, "value_max": high.value, "value_status": "range",
            "context_item_ids": list(dict.fromkeys(low.context_item_ids + high.context_item_ids)),
        }))
    return [s for s in merged if id(s) not in dropped]


def drop_conversions(statistics: list[Statistic]) -> list[Statistic]:
    """Drops a number printed in brackets straight after another one on its page: the same quantity in
    another unit ("12 bales/ha (2.7 t/ha)", "$2070/t ($470/bale)")."""
    def unit(s: Statistic) -> str:
        return (s.unit or "").strip().lower()

    dropped = set()
    for b in statistics:
        if not unit(b) or "%" in unit(b) + b.reported_text:
            continue  # "(2.6% reduction)", "(38% neutral)" are results, not conversions
        quote = " ".join(b.quote.split())
        inner = re.escape(b.reported_text.strip())
        for a in statistics:
            if a is b or a.page != b.page or id(a) in dropped or not unit(a) or unit(a) == unit(b):
                continue
            if re.search(re.escape(a.reported_text.strip()) + r"\s*\(\s*" + inner, quote):
                dropped.add(id(b))  # by object: the model's IDs can repeat
                break
    return [s for s in statistics if id(s) not in dropped]


def drop_duplicates(statistics: list[Statistic], same: set[int],
                    pairs: list[tuple[Statistic, Statistic]]) -> list[Statistic]:
    """Removes the later number of each pair judged the same measurement; the earlier keeps both sets of links."""
    merged: dict[str, list[str]] = {}
    dropped: set[str] = set()
    for k in sorted(same):
        earlier, later = pairs[k - 1]
        if earlier.statistic_id in dropped:
            continue
        dropped.add(later.statistic_id)
        ids = merged.get(earlier.statistic_id, earlier.context_item_ids)
        merged[earlier.statistic_id] = list(dict.fromkeys(ids + later.context_item_ids))
    return [s.model_copy(update={"context_item_ids": merged[s.statistic_id]}) if s.statistic_id in merged else s
            for s in statistics if s.statistic_id not in dropped]
