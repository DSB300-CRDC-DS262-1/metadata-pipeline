"""One standard name per organisation, so the same body is recorded the same way everywhere."""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

NAMES_PATH = Path(__file__).resolve().parents[4] / "data" / "organisation_names.json"
# Only long names are matched loosely; short ones ("CSRO", "ACSA") are too easy to confuse.
OCR_MATCH_MIN_LENGTH = 20
OCR_MATCH_CUTOFF = 0.92
ORGANISATION_FIELDS = ("organisation", "publisher", "funder", "copyright_holder")
_ACRONYM = r"(?:\s*\([A-Z][A-Z ]+\))?"
_SEPARATOR = r"\s*(?:;|,|&|\band\b)\s*"
_TITLE = re.compile(
    r"^(?:(?:Hon|Dr|Prof|A/Prof|Assoc\.?\s*Prof|Associate Professor|Professor|Emeritus|Mr|Mrs|Ms|Miss)\.?\s+)+", re.I
)


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    if not NAMES_PATH.is_file():
        return {}
    names = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    return {alias.lower(): standard for standard, aliases in names.items() for alias in [standard, *aliases]}


def canonical_name(name: str) -> str:
    """The standard name for one organisation, dropping a bracketed acronym like '(CRDC)'.

    A long name a few characters away from a known one (poor OCR, 'Cotton Q--owers') takes
    the known name.
    """
    cleaned = re.sub(_ACRONYM + r"\s*$", "", name.strip())
    if cleaned.lower() in _aliases():
        return _aliases()[cleaned.lower()]
    close = get_close_matches(cleaned.lower(), [a for a in _aliases() if len(a) >= OCR_MATCH_MIN_LENGTH],
                              n=1, cutoff=OCR_MATCH_CUTOFF) if len(cleaned) >= OCR_MATCH_MIN_LENGTH else []
    return _aliases()[close[0]] if close else cleaned


def organisations_named_in(text: str) -> list[str]:
    """Standard names of every known organisation mentioned in a piece of text, in order of mention."""
    found: dict[str, int] = {}
    for alias, standard in _aliases().items():
        match = re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", text, re.I)
        if match and (standard not in found or match.start() < found[standard]):
            found[standard] = match.start()
    return sorted(found, key=found.get)


def is_known_organisation(name: str) -> bool:
    return canonical_name(name).lower() in _aliases()


def _split_known(value: str) -> list[str] | None:
    """Split a list of organisations joined by ',', ';', '&' or 'and', if every part is a known name."""
    known = sorted(_aliases(), key=len, reverse=True)
    parts, rest = [], value.strip()
    while rest:
        match = next((re.match(f"(?i:{re.escape(k)})" + _ACRONYM, rest) for k in known
                      if re.match(f"(?i:{re.escape(k)})" + _ACRONYM + f"(?={_SEPARATOR}|$)", rest)), None)
        if match is None:
            return None
        parts.append(_aliases()[rest[:len(match.group(0))].split(" (")[0].lower()])
        rest = re.sub(r"^" + _SEPARATOR, "", rest[match.end():])
    return parts


def canonical_names(value: str) -> str:
    """Standardise organisations into a '; '-separated list of standard names.

    A part naming exactly one known organisation becomes that organisation, so sub-units
    ("Faculty of Science, The University of Sydney") are recorded at the top level.
    """
    known = _split_known(value)
    if known is not None:
        return "; ".join(dict.fromkeys(known))
    parts = []
    for part in value.split(";"):
        if part.strip():
            named = organisations_named_in(part)
            parts.append(named[0] if len(named) == 1 else canonical_name(part))
    return "; ".join(dict.fromkeys(parts))


def without_title(name: str) -> str:
    """A person's name without titles such as 'Hon.', 'Dr' or 'A/Prof'."""
    return _TITLE.sub("", name.strip())
