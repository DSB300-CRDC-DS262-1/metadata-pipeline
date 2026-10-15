"""Scoring extracted metadata against the reference file."""

from __future__ import annotations

from .names import ORGANISATION_FIELDS, canonical_names, without_title

# Free-text fields (supersedes_note, series wording) are not scored.
SCORED_FIELDS = [
    "title", "document_type", "authors", "organisation", "publisher", "funder", "copyright_holder",
    "publication_date", "doi", "keywords", "contact_emails",
    "journal_article.journal", "journal_article.volume", "journal_article.issue_year",
    "journal_article.pages", "journal_article.issn", "journal_article.submission_date",
    "journal_article.revised_date", "journal_article.acceptance_date",
]


def field_value(record: dict | None, name: str):
    """Look up a field by dotted name; a missing block (e.g. no journal_article) gives None."""
    for part in name.split("."):
        if not isinstance(record, dict):
            return None
        record = record.get(part)
    return record


def matches(field: str, extracted, expected) -> bool:
    """Score one field. A reference of {"accept": [...]} lists every acceptable answer."""
    if isinstance(expected, dict) and "accept" in expected:
        return any(matches(field, extracted, option) for option in expected["accept"])
    if field in ORGANISATION_FIELDS and isinstance(extracted, str) and isinstance(expected, str):
        extracted, expected = canonical_names(extracted), canonical_names(expected)
    if field == "authors" and isinstance(extracted, list) and isinstance(expected, list):
        extracted, expected = [without_title(a) for a in extracted], [without_title(a) for a in expected]
    return same(extracted, expected)


def same(a, b) -> bool:
    """Equal ignoring case and surrounding spaces; lists and '; '-separated values ignore order."""
    if a is None or b is None:
        return a == b
    return _parts(a) == _parts(b)


def _parts(value) -> set[str]:
    items = value if isinstance(value, list) else str(value).split(";")
    return {str(item).strip().lower() for item in items if str(item).strip()}
