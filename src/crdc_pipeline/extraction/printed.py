"""Fields printed in a fixed format, read with plain rules instead of a model.

Journal articles print their history ("Received 19 May 2018 ... Accepted 21 November 2018") and
DOI in a standard layout, so these are read exactly from the first pages and override the model.
"""

from __future__ import annotations

import re
from datetime import datetime

FIRST_PAGES = 2
_DATE = r"(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})"
# Two-column layouts can put a few words of the neighbouring column between a label and its date.
_GAP = r"(?:\S+\s+){0,20}?"
HISTORY = {
    "revised_date": rf"Received\s+in\s+revised\s+form\s+{_GAP}{_DATE}",
    "submission_date": rf"Received\s+(?!in\s+revised){_GAP}{_DATE}",
    "acceptance_date": rf"Accepted\s+{_GAP}{_DATE}",
    "publication_date": rf"Available\s+online\s+{_GAP}{_DATE}",
}
# Journal running header, e.g. "Journal of Cleaner Production 212 (2019) 974-985".
JOURNAL_HEADER = re.compile(r"([A-Z][A-Za-z&,' ]{3,80}?)\s+(\d{1,4})\s+\((\d{4})\)\s+(\d+)\s*[-–]\s*(\d+)")
ISSN = re.compile(r"\b(\d{4}-\d{3}[\dX])\b")
DOI = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"<>]+)")
DATE_FORMATS = ["%d %B %Y", "%d %b %Y", "%B %d, %Y", "%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"]


def valid_issn(issn: str) -> bool:
    """ISSNs end in a mod-11 check digit, which rules out look-alikes such as '2011-2012'."""
    digits = issn.replace("-", "")
    total = sum(int(d) * w for d, w in zip(digits[:7], range(8, 1, -1)))
    check = (11 - total % 11) % 11
    return digits[7] == ("X" if check == 10 else str(check))


def iso_date(value: str | None) -> str | None:
    """A date in ISO 8601 at the precision it was written, or the value unchanged if unparseable."""
    if not value:
        return value
    text = value.strip()
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if "%d" in fmt:
            return parsed.strftime("%Y-%m-%d")
        return parsed.strftime("%Y-%m") if "%B" in fmt or "%m" in fmt else parsed.strftime("%Y")
    return value


def printed_fields(pages: list[str]) -> dict[str, str]:
    """Article history dates and the document's own DOI, where printed on the first pages."""
    text = re.sub(r"\s+", " ", " ".join(pages[:FIRST_PAGES]))
    found = {}
    for field, pattern in HISTORY.items():
        match = re.search(pattern, text)
        if match:
            found[field] = iso_date(match.group(1))
    if len(found) < 2:  # a lone "Received ..." in body text is not an article history
        found = {}
    doi = DOI.search(text)
    if doi:
        found["doi"] = doi.group(1).rstrip(".,;)")
    header = JOURNAL_HEADER.search(text)
    if header and doi:  # only trust a citation-style header on a page that also carries a DOI
        journal, volume, year, first, last = header.groups()
        found.update(journal=journal.strip(), volume=volume, issue_year=year, pages=f"{first}-{last}")
        issn = next((m.group(1) for m in ISSN.finditer(text) if valid_issn(m.group(1))), None)
        if issn:
            found["issn"] = issn
    return found
