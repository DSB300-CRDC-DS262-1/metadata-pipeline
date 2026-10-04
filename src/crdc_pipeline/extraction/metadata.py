"""Document metadata schema, prompt, and extraction."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .backends import Backend
from .names import (
    ORGANISATION_FIELDS,
    canonical_names,
    is_known_organisation,
    organisations_named_in,
    without_title,
)
from .printed import iso_date, printed_fields

# A quote backing these fields names every body that holds the role, so all of them are kept.
ROLE_STATEMENTS = {
    "funder": re.compile(
        r"financial (?:contribution|support)|funded by|funding|commissioned|undertaken for|grant", re.I
    ),
    "copyright_holder": re.compile(r"©|copyright", re.I),
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
never bodies thanked for data, assistance, access or comments.
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
    metadata.authors = [without_title(a) for a in metadata.authors]
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
    _complete_from_quotes(metadata)
    metadata.document_type = _document_type(metadata)
    if not metadata.organisation and metadata.authors and all(is_known_organisation(a) for a in metadata.authors):
        metadata.organisation = canonical_names("; ".join(metadata.authors))  # a corporate author did the work
    return metadata


def _complete_from_quotes(metadata: DocumentMetadata) -> None:
    """Add every known body named in the model's own funding or copyright quote.

    Models sometimes stop after the first name ("thank X and Y for their financial contribution"
    giving only X). Quotes without the role's wording (thanks for data or comments) are ignored.
    """
    for field, statement in ROLE_STATEMENTS.items():
        named = [
            name
            for item in metadata.evidence
            if item.field == field and statement.search(item.quote)
            for name in organisations_named_in(item.quote)
        ]
        if named:
            current = [p.strip() for p in (getattr(metadata, field) or "").split(";") if p.strip()]
            setattr(metadata, field, "; ".join(dict.fromkeys(current + named)))


def _document_type(metadata: DocumentMetadata) -> str | None:
    """Lower case, and the most specific type named in the model's own quote ("Survey Report", not "report")."""
    answer = metadata.document_type.strip().lower() if metadata.document_type else None
    quotes = " ".join(e.quote for e in metadata.evidence if e.field == "document_type").lower()
    named = [t for t in DOCUMENT_TYPES if re.search(rf"\b{t}\b", quotes)]
    specific = [t for t in named if answer is None or (t != answer and t.endswith(answer))]
    return max(specific, key=len) if specific else answer


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
