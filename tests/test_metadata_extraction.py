"""Prompt, schema, and page-selection tests; the live Gemini test needs an API key."""

import os

import pytest

from crdc_pipeline.extraction.metadata import DocumentMetadata, build_context_text, build_prompt


def test_document_metadata_has_separate_organisation_publisher_funder_fields() -> None:
    metadata = DocumentMetadata(
        title="A Title",
        organisation="A University",
        publisher="A Publisher",
        funder="A Funder",
    )
    assert metadata.organisation == "A University"
    assert metadata.publisher == "A Publisher"
    assert metadata.funder == "A Funder"


def test_journal_details_only_exist_for_journal_articles() -> None:
    report = DocumentMetadata(title="A Report", publication_date="2019")
    assert report.publication_date == "2019"
    assert report.journal_article is None

    article = DocumentMetadata(title="A Paper", journal_article={"journal": "J", "submission_date": "2018-05-19"})
    assert article.journal_article.journal == "J"
    assert article.journal_article.acceptance_date is None


def test_short_document_uses_all_pages_with_page_markers() -> None:
    pages = ["page one", "page two"]
    assert build_context_text(pages) == "=== page 1 ===\npage one\n\n=== page 2 ===\npage two"


def test_reference_list_is_left_out() -> None:
    pages = ["cover", "body\nReferences\n1. Lower T. Older report. 2019.", "more citations"]
    result = build_context_text(pages)
    assert "body" in result
    assert "Older report" not in result
    assert "more citations" not in result


def test_middle_pages_contribute_funding_lines_only() -> None:
    pages = ["cover", "contents", "intro text\nWe thank CRDC for their financial contribution.\nmore",
             "results only", "last page"]
    result = build_context_text(pages)
    assert "=== page 3 (excerpt) ===" in result
    assert "financial contribution" in result
    assert "results only" not in result
    assert "=== page 5 ===" in result


def test_long_document_uses_leading_and_trailing_pages_only() -> None:
    pages = [f"page {i}" for i in range(1, 11)]
    result = build_context_text(pages)
    assert "page 1" in result
    assert "page 2" in result
    assert "page 10" in result
    assert "page 5" not in result
    assert "page 9" not in result


def test_prompt_includes_rules_and_selected_text() -> None:
    prompt = build_prompt(["Title Page Text"])
    assert "Title Page Text" in prompt
    assert "return null" in prompt
    assert "Do not fabricate a DOI" in prompt
    assert "Record email addresses only in contact_emails" in prompt
    assert "evidence entry" in prompt


def test_metadata_has_new_fields_and_evidence() -> None:
    metadata = DocumentMetadata(
        title="T",
        journal_article={"revised_date": "null"},
        evidence=[{"field": "title", "page": 1, "quote": "T", "basis": "stated"}],
    )
    assert metadata.journal_article.revised_date is None
    assert metadata.evidence[0].basis == "stated"


def test_tidy_standardises_names_and_strips_the_doi_link() -> None:
    from crdc_pipeline.extraction.metadata import tidy

    metadata = tidy(DocumentMetadata(
        funder="NSW Department of Primary Industries (NSW DPI), Australian Cotton Research and "
               "Development Corporation (CRDC)",
        doi="https://doi.org/10.1016/j.jclepro.2018.11.190",
    ))
    assert metadata.funder == "NSW Department of Primary Industries; Cotton Research and Development Corporation"
    assert metadata.doi == "10.1016/j.jclepro.2018.11.190"


def test_unknown_organisation_lists_are_left_alone() -> None:
    from crdc_pipeline.extraction.names import canonical_names

    assert canonical_names("AgHealth Australia, School of Rural Health, The University of Sydney") == (
        "AgHealth Australia, School of Rural Health, The University of Sydney"
    )
    assert canonical_names("Cotton Australia and Cotton Research and Development Corporation") == (
        "Cotton Australia; Cotton Research and Development Corporation"
    )


def test_reference_can_list_acceptable_answers() -> None:
    from crdc_pipeline.extraction.scoring import matches

    expected = {"accept": ["AgHealth Australia (The University of Sydney)", "AgHealth Australia"]}
    assert matches("organisation", "AgHealth Australia", expected)
    assert not matches("organisation", "The University of Sydney", expected)
    assert matches("publisher", None, {"accept": [None, "AgHealth Australia"]})


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY/GOOGLE_API_KEY set -- skipping live API call",
)
def test_live_extraction_on_aghealth2019() -> None:
    from pathlib import Path

    from crdc_pipeline.extraction import extract_metadata, make_backend
    from crdc_pipeline.ingestion.loader import load_document

    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs" / "aghealth2019.pdf"
    if not pdf_path.is_file():
        pytest.skip("data/raw_pdfs/aghealth2019.pdf not found")

    document = load_document(pdf_path, document_id="doc_aghealth2019")
    pages = [p.text for p in document.pages]

    result = extract_metadata(pages, make_backend("gemini"))
    metadata = result.metadata

    assert metadata.title is not None
    assert "cotton" in metadata.title.lower()
    assert any("lower" in a.lower() for a in metadata.authors)


def test_words_meaning_empty_become_none() -> None:
    metadata = DocumentMetadata(title="T", organisation="null", doi=" None ", keywords=["LCA", "n/a"])
    assert metadata.organisation is None
    assert metadata.doi is None
    assert metadata.keywords == ["LCA"]


def test_corporate_author_fills_an_empty_organisation() -> None:
    from crdc_pipeline.extraction.metadata import tidy

    assert tidy(DocumentMetadata(authors=["Voconiq"])).organisation == "Voconiq"
    assert tidy(DocumentMetadata(authors=["Jane Citizen"])).organisation is None


def test_keywords_kept_only_when_the_document_prints_them() -> None:
    from crdc_pipeline.extraction.metadata import tidy

    invented = tidy(DocumentMetadata(keywords=["Cotton Farms"]), ["A report cover", "Introduction"])
    printed = tidy(DocumentMetadata(keywords=["Life cycle assessment"]), ["Keywords: Life cycle assessment"])
    assert invented.keywords == []
    assert printed.keywords == ["Life cycle assessment"]


def test_funders_named_in_the_funding_quote_are_all_kept() -> None:
    from crdc_pipeline.extraction.metadata import tidy

    quote = ("We would like to thank the NSW Department of Primary Industries (NSW DPI) and Australian "
             "Cotton Research and Development Corporation (CRDC) for their financial contribution")
    cut_short = DocumentMetadata(
        funder="NSW Department of Primary Industries",
        evidence=[{"field": "funder", "page": 11, "quote": quote, "basis": "stated"}],
    )
    assert tidy(cut_short).funder == (
        "NSW Department of Primary Industries; Cotton Research and Development Corporation"
    )


def test_bodies_thanked_for_help_are_not_made_funders() -> None:
    from crdc_pipeline.extraction.metadata import tidy

    thanks = DocumentMetadata(
        funder="Cotton Research and Development Corporation",
        evidence=[{"field": "funder", "page": 2, "basis": "stated",
                   "quote": "we thank Cotton Australia for assistance with the data"}],
    )
    assert tidy(thanks).funder == "Cotton Research and Development Corporation"


def test_copyright_holders_named_in_the_copyright_line_are_all_kept() -> None:
    from crdc_pipeline.extraction.metadata import tidy

    cut_short = DocumentMetadata(
        copyright_holder="Cotton Australia",
        evidence=[{"field": "copyright_holder", "page": 2, "basis": "stated",
                   "quote": "© Cotton Australia and Cotton Research and Development Corporation"}],
    )
    assert tidy(cut_short).copyright_holder == "Cotton Australia; Cotton Research and Development Corporation"
