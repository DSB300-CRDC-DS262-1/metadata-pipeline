"""Finds glyphs whose embedded font labels them with the wrong character, and learns the right one.

A mislabelled glyph shows up as a single character set in a different font from both of its
neighbours inside one word or number (e.g. "974e985", where the "e" is really an en dash). The
real character is read from an image of that word, and once two reads agree the answer is saved
as a rule for that font, so later documents need no model at all.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
# Neighbours count as part of the same phrase when closer than half the font size (one space).
NEAR_GAP_RATIO = 0.5
SAME_LINE = 2.0
MAX_WORD_SIDE = 4
SAMPLES_PER_GLYPH = 5
VOTES_NEEDED = 2
# A "space" this much wider than the font's usual space is a glyph with no text mapping.
WIDE_SPACE_RATIO = 1.6

Crop = Callable[[tuple[float, float, float, float]], bytes]
ReadText = Callable[[bytes], str]


def hidden_glyphs(chars: list[dict]) -> int:
    """Glyphs drawn on the page whose text maps to a space, e.g. Voconiq's "7" in "73%"."""
    spaces: dict[tuple, list[dict]] = defaultdict(list)
    for char in chars:
        if char["text"] == " ":
            spaces[(char["fontname"], round(char["size"], 1))].append(char)
    count = 0
    for group in spaces.values():
        typical = statistics.median(c["x1"] - c["x0"] for c in group)
        count += sum(1 for c in group if c["x1"] - c["x0"] > WIDE_SPACE_RATIO * typical)
    return count


def font_key(char: dict) -> str:
    return char["fontname"].split("+")[-1]


def _family(char: dict) -> str:
    return font_key(char).split("-")[0].split(",")[0]


def _rule_key(char: dict) -> str:
    return f"{font_key(char)}|{char['text']}"


def _touching(left: dict, right: dict) -> bool:
    gap = right["x0"] - left["x1"]
    return abs(left["top"] - right["top"]) < SAME_LINE and -1.0 < gap < NEAR_GAP_RATIO * right.get("size", 10.0)


class SymbolRules:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rules: dict[str, dict] = {}
        if path is not None and path.is_file():
            self.rules = json.loads(path.read_text(encoding="utf-8"))

    def replacement(self, char: dict) -> str | None:
        votes = self.rules.get(_rule_key(char), {}).get("votes", {})
        if not votes:
            return None
        best, count = Counter(votes).most_common(1)[0]
        return best if count >= VOTES_NEEDED else None

    def apply(self, chars: list[dict]) -> None:
        """Replace glyph text using settled rules, marking the characters it changes."""
        for char in chars:
            if char.get("repaired"):
                continue
            fixed = self.replacement(char)
            if fixed is not None:
                char["original_text"] = char["text"]
                char["text"] = fixed
                char["repaired"] = True

    def suspicious(self, chars: list[dict]) -> list[int]:
        """Indexes of unrepaired glyphs set in a different font from both touching neighbours."""
        visible = [i for i, c in enumerate(chars) if c["text"].strip()]
        found = []
        for a, b, c in zip(visible, visible[1:], visible[2:]):
            left, mid, right = chars[a], chars[b], chars[c]
            if mid.get("repaired") or mid["text"] in LIGATURES:
                continue
            if mid["text"].startswith("(cid:") or mid["text"] == "\ufffd":
                found.append(b)
                continue
            if _touching(left, mid) and _touching(mid, right) and _family(left) == _family(right) != _family(mid):
                found.append(b)
        return found

    def learn(self, chars: list[dict], crop: Crop, read_text: ReadText, source: str) -> None:
        """Read word images for unsettled glyph types and vote on the real character."""
        flagged = self.suspicious(chars)
        flagged_set = set(flagged)
        samples: Counter[str] = Counter()
        for index in flagged:
            key = _rule_key(chars[index])
            if self.replacement(chars[index]) is not None or samples[key] >= SAMPLES_PER_GLYPH:
                continue
            samples[key] += 1
            word = _word_around(chars, index)
            expected, position = _expected(chars, word, flagged_set, index)
            seen = list(read_text(crop(_bbox(chars, word))).strip().strip("\"'`").replace(" ", ""))
            answer = _answer_for(expected, seen, position)
            if answer is None:
                continue
            rule = self.rules.setdefault(key, {"votes": {}, "example": "", "source": source})
            rule["votes"][answer] = rule["votes"].get(answer, 0) + 1
            if not rule["example"]:
                text_layer = "".join(chars[i]["text"] for i in word)
                rule["example"] = f"{text_layer} -> {''.join(seen)}"
        self.save()

    def save(self) -> None:
        if self.path is not None:
            self.path.write_text(json.dumps(self.rules, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _word_around(chars: list[dict], index: int) -> list[int]:
    """Indexes of the touching, same-line glyphs either side of index."""
    visible = [i for i, c in enumerate(chars) if c["text"].strip()]
    at = visible.index(index)
    start = end = at
    while start > 0 and at - start < MAX_WORD_SIDE and _touching(chars[visible[start - 1]], chars[visible[start]]):
        start -= 1
    while end + 1 < len(visible) and end - at < MAX_WORD_SIDE and _touching(chars[visible[end]], chars[visible[end + 1]]):
        end += 1
    return visible[start:end + 1]


def _expected(
    chars: list[dict], word: list[int], flagged: set[int], target: int
) -> tuple[list[str | None], int]:
    """The word as the text layer has it (None marks a glyph to be read), and target's position."""
    expected: list[str | None] = []
    position = 0
    for i in word:
        if i == target:
            position = len(expected)
        text = chars[i]["text"]
        if i in flagged:
            expected.append(None)
        else:
            expected.extend(LIGATURES.get(text, text))
    return expected, position


def _answer_for(expected: list[str | None], seen: list[str], position: int) -> str | None:
    """The character read at the flagged position, found by aligning the read with the text layer.

    Accepted only when the read lines up one-for-one there and the characters either side of
    the flagged position match, so a misread elsewhere in the phrase does not matter.
    """
    unknown = "\x00"
    left = [unknown if e is None else e for e in expected]
    matcher = SequenceMatcher(None, left, seen, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i1 <= position < i2:
            if tag != "replace" or i2 - i1 != j2 - j1:
                return None
            j = j1 + position - i1
            before_ok = position == 0 or (j > 0 and left[position - 1] in (unknown, seen[j - 1]))
            after_ok = position == len(left) - 1 or (
                j + 1 < len(seen) and left[position + 1] in (unknown, seen[j + 1])
            )
            return seen[j] if before_ok and after_ok else None
    return None


def _bbox(chars: list[dict], word: list[int]) -> tuple[float, float, float, float]:
    pad = 2.0
    return (
        min(chars[i]["x0"] for i in word) - pad,
        min(chars[i]["top"] for i in word) - pad,
        max(chars[i]["x1"] for i in word) + pad,
        max(chars[i]["bottom"] for i in word) + pad,
    )
