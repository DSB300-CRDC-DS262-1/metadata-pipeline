"""One standard name per organisation, so the same body is recorded the same way everywhere."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

NAMES_PATH = Path(__file__).resolve().parents[3] / "data" / "organisation_names.json"
ORGANISATION_FIELDS = ("organisation", "publisher", "funder", "copyright_holder")
_ACRONYM = r"(?:\s*\([A-Z][A-Z ]+\))?"
_SEPARATOR = r"\s*(?:;|,|&|\band\b)\s*"


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    if not NAMES_PATH.is_file():
        return {}
    names = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    return {alias.lower(): standard for standard, aliases in names.items() for alias in [standard, *aliases]}


def canonical_name(name: str) -> str:
    """The standard name for one organisation, dropping a bracketed acronym like '(CRDC)'."""
    cleaned = re.sub(_ACRONYM + r"\s*$", "", name.strip())
    return _aliases().get(cleaned.lower(), cleaned)


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
    """Standardise organisations into a '; '-separated list of standard names."""
    known = _split_known(value)
    if known is not None:
        return "; ".join(known)
    return "; ".join(canonical_name(part) for part in value.split(";") if part.strip())
