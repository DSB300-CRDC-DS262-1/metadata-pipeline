"""Document metadata schema, prompt, and extraction."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..backends import Backend
from .names import (
    ORGANISATION_FIELDS,
    canonical_names,
    is_known_organisation,
    organisations_named_in,
    without_title,
)
from .printed import citation, copyright_year, iso_date, printed_emails, printed_fields, published_date

# A quote backing these fields names every body that holds the role, so all of them are kept.
ROLE_STATEMENTS = {
    "funder": re.compile(
        r"financial (?:contribution|support)|funded by|funding|commissioned|undertaken for|grant", re.I
    ),
    "copyright_holder": re.compile(r"©|copyright", re.I),
    "organisation": re.compile(r"collaborat|in partnership|jointly|developed by|prepared by", re.I),
}
KEYWORDS_LABEL = re.compile(r"\bkey\s*words?\b", re.I)
DOCUMENT_TYPES = ("report", "brief", "journal article", "survey report", "conference paper", "other")
JOURNAL_DATES = ("submission_date", "revised_date", "acceptance_date")

LEADING_PAGES = 2
TRAILING_PAGES = 1
EXCERPT_CHAR_LIMIT = 6000
EMPTY_WORDS = {"", "null", "none", "n/a", "not stated", "unknown"}

# Lines on other pages that usually carry funding, rights or "this updates..." statements.
EXCERPT_PATTERN = re.compile(
    r"fund|financial|commission|undertaken for|acknowledg|copyright|update|revis|replac|supersed",
    re.I,
)


class _Cleaned(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def _empty_words_to_none(cls, value):
        """Models sometimes write the word "null" instead of leaving a field empty."""
        if isinstance(value, str) and value.strip().lower() in EMPTY_WORDS:
            return None
        if isinstance(value, str) and value.strip().startswith("["):
            try:  # a list written as JSON text where one string was asked for
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if isinstance(parsed, list) and all(isinstance(v, str) for v in parsed):
                return "; ".join(parsed) or None
        if isinstance(value, list):
            return [v for v in value if not (isinstance(v, str) and v.strip().lower() in EMPTY_WORDS)]
        return value


class JournalArticle(_Cleaned):
    journal: str | None = Field(default=None, description="Journal name.")
    volume: str | None = Field(default=None, description="Journal volume.")
    issue_year: str | None = Field(default=None, description="Year of the journal issue.")
    pages: str | None = Field(default=None, description="Page range, e.g. '974-985'.")
    issn: str | None = Field(default=None, description="ISSN, if printed.")
    submission_date: str | None = Field(default=None, description="'Received' date. ISO 8601.")
    revised_date: str | None = Field(default=None, description="'Received in revised form' date. ISO 8601.")
    acceptance_date: str | None = Field(default=None, description="'Accepted' date. ISO 8601.")


class FieldEvidence(BaseModel):
    field: str = Field(description="Name of the metadata field this supports.")
    page: int = Field(description="Page number from the '=== page N ===' marker.")
    quote: str = Field(description="Short exact quote from that page.")
    basis: Literal["stated", "inferred"] = Field(
        description="'stated' if the page says it directly, 'inferred' if deduced from it."
    )


class DocumentMetadata(_Cleaned):
    title: str | None = Field(
        default=None, description="The document's full title, as printed on the cover or first page."
    )
    document_type: str | None = Field(
        default=None,
        description="One of: " + ", ".join(DOCUMENT_TYPES) + ".",
    )
    authors: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Author names as printed, in order. If the citation credits an organisation "
        "as author (e.g. 'Voconiq (2025)'), list that organisation. Never list contact people "
        "or people only thanked in acknowledgements.",
    )
    organisation: str | None = Field(
        default=None,
        description="Who did the work: every institution the authors belong to, at the top level "
        "(a university or government department, not its faculties or research sites), separated "
        "by '; '. If the document is credited to an organisation rather than people, that "
        "organisation. Never the publisher, a funder or the body the work was undertaken for, "
        "unless an author belongs to it.",
    )
    publisher: str | None = Field(
        default=None,
        description="Who published or distributes the document, only if named (e.g. a journal "
        "publisher, or the body named in the document's own citation line).",
    )
    funder: str | None = Field(
        default=None,
        description="Every body that funded, financially supported, commissioned or had the work "
        "undertaken for it, from any explicit statement anywhere in the text. Separate several "
        "with '; '.",
    )
    copyright_holder: str | None = Field(
        default=None, description="Who holds copyright, from a copyright line."
    )
    publication_date: str | None = Field(
        default=None,
        description="When the document was first made public: for a journal article the "
        "'Available online' date, otherwise the date printed on the document. ISO 8601: "
        "'YYYY-MM-DD', 'YYYY-MM' or 'YYYY', as precise as the text allows.",
    )
    journal_article: JournalArticle | None = Field(
        default=None,
        description="Only for a journal article: its journal details and review dates. Null for "
        "any other kind of document.",
    )
    doi: str | None = Field(default=None, description="DOI only, e.g. '10.1016/j.jclepro.2018.11.190'.")
    keywords: list[str] = Field(default_factory=list, max_length=20, description="Keywords, only if listed.")
    series: str | None = Field(
        default=None, description="Name of the ongoing program or report series, if any."
    )
    supersedes_note: str | None = Field(
        default=None,
        description="Quote or closely paraphrase any sentence where the document says it updates, "
        "revises or replaces an earlier report. Null if it makes no such claim.",
    )
    contact_emails: list[str] = Field(
        default_factory=list, max_length=20, description="Email addresses printed in the document."
    )
    evidence: list[FieldEvidence] = Field(
        default_factory=list,
        max_length=25,
        description="One entry for every non-empty field above, citing where it came from.",
    )


PROMPT_TEMPLATE = """You are extracting bibliographic metadata from a research document for a \
government research agency's document pipeline. Accuracy matters more than completeness: it is \
much better to return null for a field than to guess or infer a plausible-sounding value.

Rules:
- Use ONLY information present in the text below.
- If a field cannot be determined from this text, return null (or an empty list).
- Keep organisation, publisher and funder separate. The body a study was "undertaken for" is a \
funder, never the organisation. The same body can fill more than one role when the text supports it.
- funder lists every body named in a funding statement ("funding source", "financial \
contribution", "funded by") and every body the work was commissioned by or undertaken for. These \
statements are often in acknowledgements or the method, not on the cover: check the excerpts. \
When one statement names several bodies ("thank X and Y for their financial contribution"), list \
every one of them, not just the first. Only bodies credited with money or commissioning count: \
never bodies thanked for data, assistance, access or comments, never bodies that only received \
funding, and never bodies said to have funded or commissioned other studies the document mentions.
- publication_date for a journal article is the "Available online" date, not the issue year.
- Dates use ISO 8601 ('YYYY-MM-DD', 'YYYY-MM' or 'YYYY'), never the printed format.
- Record email addresses only in contact_emails, never in other fields.
- Do not fabricate a DOI. Only record one printed in the text.
- For every field you fill, add an evidence entry with the page number from the nearest \
'=== page N ===' marker, a short exact quote, and whether the value is stated or inferred.

Document text (first pages, last page, and excerpts from other pages):
---
{text}
---

Extract the metadata now."""


def _without_reference_list(pages: list[str]) -> list[tuple[int, str]]:
    """Numbered pages up to the reference list; citations of other works carry their metadata."""
    kept = []
    for n, text in enumerate(pages, start=1):
        lines = text.splitlines()
        cut = next((i for i, line in enumerate(lines) if line.strip().lower() == "references"), None)
        if cut is not None:
            if "\n".join(lines[:cut]).strip():
                kept.append((n, "\n".join(lines[:cut])))
            break
        kept.append((n, text))
    return kept


def build_context_text(pages: list[str]) -> str:
    """First two pages, last page, and keyword excerpts from the pages in between."""
    numbered = _without_reference_list(pages)
    if len(numbered) <= LEADING_PAGES + TRAILING_PAGES:
        return "\n\n".join(f"=== page {n} ===\n{text}" for n, text in numbered)

    leading = numbered[:LEADING_PAGES]
    trailing = numbered[-TRAILING_PAGES:]
    middle = numbered[LEADING_PAGES:-TRAILING_PAGES]

    parts = [f"=== page {n} ===\n{text}" for n, text in leading]
    budget = EXCERPT_CHAR_LIMIT
    for n, text in middle:
        lines = text.splitlines()
        keep = sorted({j for i, line in enumerate(lines) if EXCERPT_PATTERN.search(line)
                       for j in (i - 1, i, i + 1) if 0 <= j < len(lines)})
        excerpt = "\n".join(lines[j] for j in keep).strip()
        if excerpt and budget > 0:
            excerpt = excerpt[:budget]
            budget -= len(excerpt)
            parts.append(f"=== page {n} (excerpt) ===\n{excerpt}")
    parts += [f"=== page {n} ===\n{text}" for n, text in trailing]
    return "\n\n".join(parts)


def build_prompt(pages: list[str]) -> str:
    return PROMPT_TEMPLATE.format(text=build_context_text(pages))


def tidy(metadata: DocumentMetadata, pages: list[str] | None = None) -> DocumentMetadata:
    """Printed facts over model answers, ISO dates, standard names and a bare DOI."""
    for field, value in printed_fields(pages or []).items():
        if field in JournalArticle.model_fields:
            metadata.journal_article = metadata.journal_article or JournalArticle()
            setattr(metadata.journal_article, field, value)
        else:
            setattr(metadata, field, value)
    if pages is not None and not KEYWORDS_LABEL.search(" ".join(pages[:LEADING_PAGES])):
        metadata.keywords = []  # keywords only count when the document prints a keyword list
    metadata.authors = [BRACKETED_TAIL.sub("", without_title(a)) for a in metadata.authors]
    metadata.authors = [a for a in metadata.authors if not _only_managed(a, metadata.evidence)]
    metadata.publication_date = iso_date(metadata.publication_date)
    if metadata.journal_article:
        for field in JOURNAL_DATES:
            setattr(metadata.journal_article, field, iso_date(getattr(metadata.journal_article, field)))
    for field in ORGANISATION_FIELDS:
        value = getattr(metadata, field)
        if value:
            setattr(metadata, field, canonical_names(value))
    if metadata.doi:
        metadata.doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", metadata.doi.strip())
    _drop_unsupported_funders(metadata)
    _complete_from_quotes(metadata)
    _organisation_from_corporate_author(metadata)
    journal = metadata.journal_article.journal if metadata.journal_article else None
    if journal and metadata.publisher and _words(metadata.publisher) == _words(journal):
        metadata.publisher = metadata.copyright_holder  # the journal is not its own publisher
    _publisher_from_copyright(metadata)
    if pages is not None and journal:
        _organisations_from_affiliations(metadata, pages)
    _apply_citation(metadata, citation(pages or []))  # after names are standardised and funders completed
    if pages is not None:
        metadata.contact_emails = printed_emails(pages)  # the whole document, not only the pages the model saw
        metadata.publication_date = published_date(pages) or metadata.publication_date
        _copyright_from_print(metadata, pages)
        year = copyright_year(pages)
        if year and (metadata.publication_date or "9999")[:4] < year:
            metadata.publication_date = year  # a report for 2024 copyrighted 2025 came out in 2025
    metadata.document_type = _document_type(metadata)
    if not metadata.organisation and metadata.authors and all(is_known_organisation(a) for a in metadata.authors):
        metadata.organisation = canonical_names("; ".join(metadata.authors))  # a corporate author did the work
    if (not metadata.authors and metadata.organisation and metadata.organisation == metadata.publisher
            and ";" not in metadata.organisation):
        metadata.authors = [metadata.organisation]  # a body's own report with no named authors is credited to it
    return metadata


COPYRIGHT_LINE = re.compile(r"©|\bcopyright\b", re.I)
NOT_AUTHORSHIP = re.compile(r"\b(?:managed|coordinated|overseen)\s+by\b|\bcontact\b", re.I)


def _only_managed(author: str, evidence: list[FieldEvidence]) -> bool:
    """An author whose only quotes say they managed or coordinated the project, or are a contact."""
    quotes = [e.quote for e in evidence if e.field == "authors" and author.lower() in e.quote.lower()]
    return bool(quotes) and all(NOT_AUTHORSHIP.search(q) and not CREDITED_BY.search(q) for q in quotes)


def _copyright_from_print(metadata: DocumentMetadata, pages: list[str]) -> None:
    """Copyright holders need a printed copyright line: taken from the '©' line, or cleared without one."""
    lines = [line for text in pages for line in text.splitlines() if COPYRIGHT_LINE.search(line)]
    if not lines:
        metadata.copyright_holder = None
    elif not metadata.copyright_holder:
        named = [n for line in lines if "©" in line for n in organisations_named_in(line)]
        metadata.copyright_holder = "; ".join(dict.fromkeys(named)) or None


BRACKETED_TAIL = re.compile(r"\s*\((?![A-Z]{2,8}\))[^)]*\)\s*$")  # "(on behalf of ...)", but not "(CRDC)"
CREDITED_BY = re.compile(r"\b(?:developed|prepared|written|compiled|produced|authored)\s+by\b", re.I)
# "X and Y received funding from Z": X and Y are recipients, not funders.
RECIPIENTS = re.compile(r"^(.*?)\breceived\b(?=.*\b(?:funding|grant|support)\b.*\bfrom\b)", re.I)
# Words that mark a body rather than a person.
CORPORATE = re.compile(
    r"\b(?:Group|Committee|Association|Corporation|Council|Department|Institute|University|"
    r"Pty|Ltd|Inc|Incorporated|Australia|Board|Centre|Center|Organisation|Agency|Unit|Team)\b"
)
# A quote that is a figure or table source line, not a statement about this document.
SOURCE_LINE = re.compile(r"^\W*sources?\s*:", re.I)


def _weak(item: FieldEvidence) -> bool:
    """Inferred, a figure's source line, or (for funders) a copyright line with no word about funding."""
    if item.basis == "inferred" or SOURCE_LINE.search(item.quote):
        return True
    return (item.field == "funder" and bool(COPYRIGHT_LINE.search(item.quote))
            and not ROLE_STATEMENTS["funder"].search(item.quote))


def _drop_unsupported_funders(metadata: DocumentMetadata) -> None:
    """Remove funders backed only by a figure's "Source:" line or a quote the model marked inferred."""
    if not metadata.funder:
        return
    weak, strong = set(), set()
    quotes = [item for item in metadata.evidence if item.field == "funder"]
    for item in quotes:
        (weak if _weak(item) else strong).update(organisations_named_in(item.quote))

    def unsupported(name: str) -> bool:
        if name in weak - strong:
            return True
        # Names outside the names file: every quote that mentions them is weak.
        mentioned = [item for item in quotes if name.lower() in item.quote.lower()]
        return bool(mentioned) and all(_weak(item) for item in mentioned)

    kept = [p.strip() for p in metadata.funder.split(";") if p.strip() and not unsupported(p.strip())]
    metadata.funder = "; ".join(kept) or None


# Top-level institutions in an affiliation list; departments, centres and institutes within them are not.
INSTITUTION = re.compile(
    r"\b(?:(?:The )?University of(?: [A-Z][\w'-]*)+|(?:[A-Z][\w'-]* )+University|"
    r"United States Department of Agriculture|CSIRO|NSW Department of Primary Industries)\b"
)
AFFILIATION_END = re.compile(r"corresponding author|abstract|keywords|introduction", re.I)
AFFILIATION_REACH = 1000


def _organisations_from_affiliations(metadata: DocumentMetadata, pages: list[str]) -> None:
    """Every institution in the affiliation list the organisation quote comes from.

    Journal articles list one affiliation per author; models often stop after the first.
    """
    for item in metadata.evidence:
        if item.field != "organisation" or not 1 <= item.page <= len(pages):
            continue
        text = " ".join(pages[item.page - 1].split())
        start = text.find(" ".join(item.quote.split())[:60])
        if start < 0:
            continue
        block = text[start:start + AFFILIATION_REACH]
        end = AFFILIATION_END.search(block)
        block = block[:end.start()] if end else block
        found = [canonical_names(m.group(0)) for m in INSTITUTION.finditer(block)]
        current = [p.strip() for p in (metadata.organisation or "").split(";") if p.strip()]
        metadata.organisation = "; ".join(dict.fromkeys(current + found)) or None


def _publisher_from_copyright(metadata: DocumentMetadata) -> None:
    """A publisher that is one of several joint copyright holders publishes with the others."""
    publishers = {p.strip() for p in (metadata.publisher or "").split(";") if p.strip()}
    holders = [p.strip() for p in (metadata.copyright_holder or "").split(";") if p.strip()]
    if publishers and publishers < set(holders):
        metadata.publisher = "; ".join(holders)


def _organisation_from_corporate_author(metadata: DocumentMetadata) -> None:
    """A body credited as author ("developed by the Sustainability Working Group") did the work.

    Applies when the organisation's own quote names that author and not the organisation given,
    or when the author's quote credits it with the work ("developed by", "prepared by").
    """
    corporate = [a for a in metadata.authors if CORPORATE.search(a)]
    if len(corporate) != 1 or len(metadata.authors) != 1:
        return
    author = corporate[0]
    quotes = [e.quote.lower() for e in metadata.evidence if e.field == "organisation"]
    credits = [e.quote.lower() for e in metadata.evidence if e.field == "authors" and CREDITED_BY.search(e.quote)]
    current = [p.strip() for p in (metadata.organisation or "").split(";") if p.strip()]
    named_in_own_quote = any(author.lower() in q and not any(c.lower() in q for c in current) for q in quotes)
    if named_in_own_quote or any(author.lower() in q for q in credits):
        metadata.organisation = canonical_names(author)


PERSON_INITIALS = re.compile(r"\b[A-Z]\.?(?:\s*[A-Z]\.?)?(?:\s*(?:,|&|\band\b)|\s*$)|\bet al\b")


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower().replace("’", "'"))) - {"the"}


def _apply_citation(metadata: DocumentMetadata, cited: dict[str, str]) -> None:
    """The document's own suggested citation settles its year, publisher, title and organisation."""
    if not cited:
        return
    if (metadata.publication_date or "")[:4] != cited["year"]:
        metadata.publication_date = cited["year"]  # the citation beats a survey or data year
    if cited.get("publisher"):
        metadata.publisher = canonical_names(cited["publisher"])
    title = cited.get("title")
    if title and metadata.title and _words(metadata.title) <= _words(title):
        metadata.title = title  # the model kept only part of the title, or reworded its punctuation
    author = cited.get("author", "")
    corporate = author and not PERSON_INITIALS.search(author)
    funders = {p.strip() for p in (metadata.funder or "").split(";") if p.strip()}
    doers = {p.strip() for p in (metadata.organisation or "").split(";") if p.strip()}
    if corporate and doers <= funders:
        metadata.organisation = canonical_names(author)  # who the work was done for is the funder, not the organisation
    if corporate and not any(_words(a) <= _words(author) for a in metadata.authors):
        metadata.authors = [author]  # "Voconiq (2023)": the document credits the body, not a contact person


def _complete_from_quotes(metadata: DocumentMetadata) -> None:
    """Add every known body named in the model's own funding or copyright quote.

    Models sometimes stop after the first name ("thank X and Y for their financial contribution"
    giving only X). Quotes without the role's wording (thanks for data or comments) are ignored.
    """
    for field, statement in ROLE_STATEMENTS.items():
        named = [
            name
            for item in metadata.evidence
            if item.field == field and statement.search(item.quote) and not _weak(item)
            for name in organisations_named_in(_without_recipients(item.quote) if field == "funder" else item.quote)
        ]
        if field == "funder":
            # Only received funding, and no funding statement credits them with giving any.
            recipients = {n for item in metadata.evidence if item.field == "funder"
                          for n in organisations_named_in(_recipients(item.quote))} - set(named)
            kept = [p.strip() for p in (metadata.funder or "").split(";") if p.strip() and p.strip() not in recipients]
            metadata.funder = "; ".join(kept) or None
        if named:
            current = [p.strip() for p in (getattr(metadata, field) or "").split(";") if p.strip()]
            # A part naming several known bodies is a list the names file could not split; its names are added singly.
            current = [p for p in current if len(organisations_named_in(p)) < 2]
            setattr(metadata, field, "; ".join(dict.fromkeys(current + named)))


def _recipients(quote: str) -> str:
    found = RECIPIENTS.match(quote)
    return found.group(1) if found else ""


def _without_recipients(quote: str) -> str:
    return quote[len(_recipients(quote)):]


def _document_type(metadata: DocumentMetadata) -> str | None:
    """Lower case, and the most specific type named in the model's own quote ("Survey Report", not "report")."""
    answer = metadata.document_type.strip().lower() if metadata.document_type else None
    quotes = " ".join(e.quote for e in metadata.evidence if e.field == "document_type").lower()
    named = [t for t in DOCUMENT_TYPES if re.search(rf"\b{t}\b", quotes)]
    specific = [t for t in named if answer is None or (t != answer and t.endswith(answer))]
    answer = max(specific, key=len) if specific else answer
    if answer and answer not in DOCUMENT_TYPES:
        # e.g. "sustainability report" -> "report": the longest allowed type it contains, else "other"
        answer = max((t for t in DOCUMENT_TYPES if re.search(rf"\b{t}\b", answer)), key=len, default="other")
    if answer == "report" and re.search(r"\bsurvey\b", metadata.title or "", re.I):
        answer = "survey report"  # "Findings of the CRDC's survey of cotton growers"
    return answer


class MetadataExtractionResult(BaseModel):
    metadata: DocumentMetadata
    resolved_model: str
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


def extract_metadata(pages: list[str], backend: Backend) -> MetadataExtractionResult:
    result = backend.generate(build_prompt(pages), DocumentMetadata)
    return MetadataExtractionResult(
        metadata=tidy(result.parsed, pages),
        resolved_model=result.model,
        seconds=result.seconds,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
    )
