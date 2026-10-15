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
# "The Journal of Cotton Science 25:1-6 (2021)": a format specific enough to trust without a DOI.
JOURNAL_HEADER_COLON = re.compile(r"([A-Z][A-Za-z&,' ]{3,80}?)\s+(\d{1,4}):(\d+)\s*[-–]\s*(\d+)\s+\((\d{4})\)")
ISSN = re.compile(r"\b(\d{4}-\d{3}[\dX])\b")
DOI = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"<>]+)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
WRAPPED_DOMAIN = re.compile(r"(\S)\s*\n\s*(com\.au|com|org|au)\b")
PUBLISHED = re.compile(
    r"\bPublished\s*:?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?:19|20)\d\d|(?:19|20)\d\d)\b"
)
COPYRIGHT = re.compile(r"©\s*(?:[A-Z][\w&.,' ]{1,80}?\s+)?((?:19|20)\d\d)\b")
# A "Citation" label followed within a few words by an author and a bracketed year.
CITATION_PAGES = 3
CITATION = re.compile(r"\b(?:citation|cite as|cite this)\W{0,3}(?:\S+\s+){0,6}?\S*\(((?:19|20)\d\d)\)", re.I)
# After the year: "Title. Publisher, Place."
CITATION_REST = re.compile(r"\.?\s*(?P<title>[^.]{5,200})\.\s+(?P<publisher>[A-Z][^.,]{1,100}),\s+[A-Z][A-Za-z ]{1,40}\.")
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


def printed_emails(pages: list[str]) -> list[str]:
    """Every email address printed in the document, in order, without repeats."""
    found = []
    for text in pages:
        joined = WRAPPED_DOMAIN.sub(r"\1\2", text)  # "austcottonshippers.\ncom.au" split over a line
        found += [m.rstrip(".") for m in EMAIL.findall(joined)]
    return list(dict.fromkeys(found))


def published_date(pages: list[str]) -> str | None:
    """A date printed as "Published: October 2024", anywhere in the document."""
    match = next((m for text in pages for m in [PUBLISHED.search(text)] if m), None)
    return iso_date(match.group(1)) if match else None


def copyright_year(pages: list[str]) -> str | None:
    """The latest year in a "© Holder 2025" or "© 2018 Holder" line, anywhere in the document."""
    years = [m.group(1) for text in pages for m in COPYRIGHT.finditer(text)]
    return max(years, default=None)


def citation_year(pages: list[str]) -> str | None:
    """The year in a suggested citation ("Citation: Roth Rural (2017) ..."), from the first pages."""
    return citation(pages).get("year")


def citation(pages: list[str]) -> dict[str, str]:
    """Year, author, title and publisher of the document's own suggested citation, where printed.

    "Citation Roth Rural (2017) Cotton Growing Practices 2016: Findings ... growers. Cotton Research
    and Development Corporation, Narrabri." gives all four; parts that do not fit are left out.
    """
    text = re.sub(r"\s+", " ", " ".join(pages[:CITATION_PAGES]))
    match = CITATION.search(text)
    if not match:
        return {}
    found = {"year": match.group(1)}
    author = re.sub(r"^\W*(?:citation|cite as|cite this)\W*", "", text[match.start():match.end()], flags=re.I)
    found["author"] = author[: author.rfind("(")].strip(" ,.")
    rest = CITATION_REST.match(text[match.end():])
    if rest:
        found["title"] = rest.group("title").strip()
        found["publisher"] = rest.group("publisher").strip()
    return found


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
    colon = JOURNAL_HEADER_COLON.search(text)
    if header and doi:  # only trust a citation-style header on a page that also carries a DOI
        journal, volume, year, first, last = header.groups()
        found.update(journal=journal.strip(), volume=volume, issue_year=year, pages=f"{first}-{last}")
    elif colon:
        journal, volume, first, last, year = colon.groups()
        found.update(journal=journal.strip(), volume=volume, issue_year=year, pages=f"{first}-{last}")
    if "journal" in found:
        issn = next((m.group(1) for m in ISSN.finditer(text) if valid_issn(m.group(1))), None)
        if issn:
            found["issn"] = issn
    return found
