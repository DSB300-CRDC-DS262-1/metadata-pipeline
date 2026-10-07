"""Scoring extracted context items and links against data/reference/contexts.json."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher


ITEM_OVERLAP = 0.6  # share of the shorter quote two context items must share to state the same fact


def same_fact(reference: dict, extracted: dict) -> bool:
    """Context items whose quotes largely overlap, on any page (abstracts repeat methods text).

    A reference item can list other sentences stating the same fact in `also_stated`.
    """
    quotes = [reference["quote"]] + [q["quote"] for q in reference.get("also_stated", [])]
    return any(_quotes_overlap(q, extracted.get("quote", "")) for q in quotes)


def _quotes_overlap(first: str, second: str) -> bool:
    a = " ".join(first.lower().split())
    b = " ".join(second.lower().split())
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    overlap = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size
    return overlap >= ITEM_OVERLAP * min(len(a), len(b))


# A quote the model reworded ("Safe Work Australia Australia", "Earlier data costs (2012 dollars) were updated to
# 2018 Australian dollars") still states the reference fact if it shares most of its key words with it.
KEY_WORDS_SHARED = 0.7
COMMON_WORDS = set(
    "the a an of to in for and or is are be was were been that this these those it its on by with as at from their they "
    "our we you not can all have has had which who also than then there such into over under more most other each any per"
    .split())


def key_words(text: str) -> set[str]:
    """Words that carry the fact: no common words, short words or plural endings; numbers kept ("2018", "1.5")."""
    found = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return {w.rstrip("s") if len(w) > 4 else w for w in found if w not in COMMON_WORDS and len(w) > 2}


def _reworded_match(reference_items: list[dict], extracted: dict) -> set[str]:
    """The reference item sharing most of the extracted quote's key words, weighted so rarer words count more.

    Words found in many of the document's reference quotes ("cotton", "data") say little about which fact it is.
    """
    words = key_words(extracted.get("quote", ""))
    if not words or not reference_items:
        return set()
    stated = {r["item_id"]: key_words(" ".join([r["quote"], *(q["quote"] for q in r.get("also_stated", []))]))
              for r in reference_items}
    count = Counter(w for ws in stated.values() for w in ws)
    weight = {w: math.log((len(stated) + 1) / (count[w] + 1)) + 1 for w in words}
    total = sum(weight.values())
    best, share = max(((item_id, sum(weight[w] for w in words & ws) / total) for item_id, ws in stated.items()),
                      key=lambda pair: pair[1])
    return {best} if share >= KEY_WORDS_SHARED else set()


def item_map(reference_items: list[dict], extracted_items: list[dict]) -> dict[str, set[str]]:
    """Each extracted item ID mapped to the reference items stating the same fact: by overlapping quotes, or
    failing that, by the key words of a reworded quote."""
    mapping = {}
    for e in extracted_items:
        same = {r["item_id"] for r in reference_items if same_fact(r, e)}
        mapping[e["item_id"]] = same or _reworded_match(reference_items, e)
    return mapping


def context_links(pairs: list[tuple[dict, dict]], reference: dict, extracted: dict) -> tuple[int, int, int, int]:
    """(needed facts found, needed facts, links given, right links) over matched statistics.

    Recall is counted by fact: needed facts found / needed facts. Precision is counted by link: a link is
    right when the fact it states is needed for the statistic or acceptable for it (a reference
    statistic's `acceptable_item_ids`, e.g. a model assumption behind a result, are fair to link but not
    needed); a second link to the same right fact is redundant, not wrong.
    """
    mapping = item_map(reference["context_items"], extracted["context_items"])
    whole_ref = [i["item_id"] for i in reference["context_items"] if i.get("whole_document")]
    whole_ext = [i["item_id"] for i in extracted["context_items"] if i.get("whole_document")]
    found = wanted = given = right = 0
    for ref, ext in pairs:
        expected = set(ref.get("context_item_ids", [])) | set(whole_ref)
        fair = expected | set(ref.get("acceptable_item_ids", []))
        linked = set(ext.get("context_item_ids", [])) | set(whole_ext)
        covered = set().union(*(mapping.get(i, set()) for i in linked)) if linked else set()
        wanted += len(expected)
        given += len(linked)
        found += len(expected & covered)
        right += sum(bool(mapping.get(i, set()) & fair) for i in linked)
    return found, wanted, given, right


KEY_CONDITIONS = ("data_source", "sample_population", "geographic_region", "time_period", "currency_basis")


def key_conditions(pairs: list[tuple[dict, dict]], reference: dict, extracted: dict) -> dict[str, list[int]]:
    """Per key kind of condition: [right, wrong, missing] over matched statistics whose reference has one.

    right: the extraction links the same fact. wrong: it links only facts of that kind that belong
    elsewhere (e.g. a different data source). missing: it links nothing of that kind.
    """
    mapping = item_map(reference["context_items"], extracted["context_items"])
    ref_items = {i["item_id"]: i for i in reference["context_items"]}
    ext_items = {i["item_id"]: i for i in extracted["context_items"]}
    whole_ref = [i["item_id"] for i in reference["context_items"] if i.get("whole_document")]
    whole_ext = [i["item_id"] for i in extracted["context_items"] if i.get("whole_document")]
    counts = {kind: [0, 0, 0] for kind in KEY_CONDITIONS}
    for ref, ext in pairs:
        expected = set(ref.get("context_item_ids", [])) | set(whole_ref)
        linked = [i for i in set(ext.get("context_item_ids", [])) | set(whole_ext) if i in ext_items]
        for kind in KEY_CONDITIONS:
            wanted = {i for i in expected if ref_items[i]["item_type"] == kind}
            if not wanted:
                continue
            given = [i for i in linked if ext_items[i]["item_type"] == kind]
            facts = set().union(*(mapping.get(i, set()) for i in given)) if given else set()
            if facts & wanted:
                counts[kind][0] += 1
            elif facts or given:
                counts[kind][1] += 1
            else:
                counts[kind][2] += 1
    return counts
