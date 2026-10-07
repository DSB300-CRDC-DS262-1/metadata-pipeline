"""Scoring extracted context items and links against data/reference/contexts.json."""

from __future__ import annotations

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


def item_map(reference_items: list[dict], extracted_items: list[dict]) -> dict[str, set[str]]:
    """Each extracted item ID mapped to the reference items stating the same fact."""
    return {
        e["item_id"]: {r["item_id"] for r in reference_items if same_fact(r, e)}
        for e in extracted_items
    }


def context_links(pairs: list[tuple[dict, dict]], reference: dict, extracted: dict) -> tuple[int, int, int]:
    """(correct links, reference links, extracted links) over matched statistics, by specific fact."""
    mapping = item_map(reference["context_items"], extracted["context_items"])
    whole_ref = [i["item_id"] for i in reference["context_items"] if i.get("whole_document")]
    whole_ext = [i["item_id"] for i in extracted["context_items"] if i.get("whole_document")]
    correct = wanted = given = 0
    for ref, ext in pairs:
        expected = set(ref.get("context_item_ids", [])) | set(whole_ref)
        linked = set(ext.get("context_item_ids", [])) | set(whole_ext)
        covered = set().union(*(mapping.get(i, set()) for i in linked)) if linked else set()
        wanted += len(expected)
        given += len(linked)
        correct += len(expected & covered)
    return correct, wanted, given


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
