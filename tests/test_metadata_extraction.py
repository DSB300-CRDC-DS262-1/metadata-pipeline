"""Prompt, schema, and page-selection tests; the live Gemini test needs an API key."""

import os

import pytest

from crdc_pipeline.extraction.metadata.extract import DocumentMetadata, build_context_text, build_prompt


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
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(
        funder="NSW Department of Primary Industries (NSW DPI), Australian Cotton Research and "
               "Development Corporation (CRDC)",
        doi="https://doi.org/10.1016/j.jclepro.2018.11.190",
    ))
    assert metadata.funder == "NSW Department of Primary Industries; Cotton Research and Development Corporation"
    assert metadata.doi == "10.1016/j.jclepro.2018.11.190"


def test_unknown_organisation_lists_are_left_alone() -> None:
    from crdc_pipeline.extraction.metadata.names import canonical_names

    assert canonical_names("AgHealth Australia, School of Rural Health, The University of Sydney") == (
        "AgHealth Australia, School of Rural Health, The University of Sydney"
    )
    assert canonical_names("Cotton Australia and Cotton Research and Development Corporation") == (
        "Cotton Australia; Cotton Research and Development Corporation"
    )


def test_reference_can_list_acceptable_answers() -> None:
    from crdc_pipeline.extraction.metadata.scoring import matches

    expected = {"accept": ["AgHealth Australia (The University of Sydney)", "AgHealth Australia"]}
    assert matches("organisation", "AgHealth Australia", expected)
    assert not matches("organisation", "The University of Sydney", expected)
    assert matches("publisher", None, {"accept": [None, "AgHealth Australia"]})


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY/GOOGLE_API_KEY set, skipping live API call",
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
    from crdc_pipeline.extraction.metadata.extract import tidy

    assert tidy(DocumentMetadata(authors=["Voconiq"])).organisation == "Voconiq"
    assert tidy(DocumentMetadata(authors=["Jane Citizen"])).organisation is None


def test_authors_are_recorded_without_titles() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    assert tidy(DocumentMetadata(authors=["Hon. A/Prof Tony Lower", "Drew Miles"])).authors == [
        "Tony Lower", "Drew Miles"
    ]


def test_document_type_takes_the_most_specific_type_in_its_quote() -> None:
    from crdc_pipeline.extraction.metadata.extract import FieldEvidence, tidy

    quote = FieldEvidence(field="document_type", page=1, quote="2025 Survey Report", basis="stated")
    assert tidy(DocumentMetadata(document_type="Report", evidence=[quote])).document_type == "survey report"
    assert tidy(DocumentMetadata(document_type="Journal Article")).document_type == "journal article"


def test_keywords_kept_only_when_the_document_prints_them() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    invented = tidy(DocumentMetadata(keywords=["Cotton Farms"]), ["A report cover", "Introduction"])
    printed = tidy(DocumentMetadata(keywords=["Life cycle assessment"]), ["Keywords: Life cycle assessment"])
    assert invented.keywords == []
    assert printed.keywords == ["Life cycle assessment"]


def test_funders_named_in_the_funding_quote_are_all_kept() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

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
    from crdc_pipeline.extraction.metadata.extract import tidy

    thanks = DocumentMetadata(
        funder="Cotton Research and Development Corporation",
        evidence=[{"field": "funder", "page": 2, "basis": "stated",
                   "quote": "we thank Cotton Australia for assistance with the data"}],
    )
    assert tidy(thanks).funder == "Cotton Research and Development Corporation"


def test_copyright_holders_named_in_the_copyright_line_are_all_kept() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    cut_short = DocumentMetadata(
        copyright_holder="Cotton Australia",
        evidence=[{"field": "copyright_holder", "page": 2, "basis": "stated",
                   "quote": "© Cotton Australia and Cotton Research and Development Corporation"}],
    )
    assert tidy(cut_short).copyright_holder == "Cotton Australia; Cotton Research and Development Corporation"


def test_citation_year_replaces_a_survey_year() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    pages = ["Cotton Growing Practices 2016", "", "Citation Roth Rural (2017) Cotton Growing Practices 2016"]
    assert tidy(DocumentMetadata(publication_date="2016"), pages).publication_date == "2017"
    assert tidy(DocumentMetadata(publication_date="December 2023"), ["CITATION Voconiq (2023)"]).publication_date == "2023-12"


def test_document_types_outside_the_list_map_to_the_nearest_allowed_type() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    assert tidy(DocumentMetadata(document_type="Sustainability Report")).document_type == "report"
    assert tidy(DocumentMetadata(document_type="factsheet")).document_type == "other"


CITATION_PAGES = ["Cotton Growing Practices 2016", "Roth Rural for the CRDC",
                  "Citation Roth Rural (2017) Cotton Growing Practices 2016: Findings of the CRDC's survey of cotton "
                  "growers. Cotton Research and Development Corporation, Narrabri. Disclaimer"]


def test_citation_sets_publisher_full_title_and_who_did_the_work() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(
        title="Findings of CRDC's Survey of Cotton Growers", organisation="CRDC", funder="CRDC"), CITATION_PAGES)
    assert metadata.title == "Cotton Growing Practices 2016: Findings of the CRDC's survey of cotton growers"
    assert metadata.publisher == "Cotton Research and Development Corporation"
    assert metadata.organisation == "Roth Rural"


def test_citation_leaves_a_separate_organisation_and_a_different_title_alone() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(title="Community Trust 2025 Survey Report", organisation="Voconiq",
                                     funder="CRDC"), CITATION_PAGES)
    assert metadata.organisation == "Voconiq"
    assert metadata.title == "Community Trust 2025 Survey Report"
    people = ["Citation Lower T, Peachey K (2019) Sentinel Health. AgHealth Australia, Dubbo."]
    assert tidy(DocumentMetadata(organisation="CRDC", funder="CRDC"), people).organisation == "Cotton Research and Development Corporation"



def test_emails_and_published_date_come_from_the_whole_document() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    pages = ["Annual Report 2023-24", "Contact crdc@crdc.com.au"] + ["Body"] * 5 + [
        "© CRDC 2024 Published: October 2024 Email: crdc@crdc.com.au or marketing@austcottonshippers.\ncom.au"]
    metadata = tidy(DocumentMetadata(publication_date="2023-24", contact_emails=[]), pages)
    assert metadata.publication_date == "2024-10"
    assert metadata.contact_emails == ["crdc@crdc.com.au", "marketing@austcottonshippers.com.au"]


def test_funders_backed_only_by_a_source_line_or_an_inferred_quote_are_dropped() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    figure = DocumentMetadata(funder="CRDC", evidence=[
        {"field": "funder", "page": 25, "quote": "Source: CRDC commissioned research.", "basis": "stated"}])
    guessed = DocumentMetadata(funder="Cotton Australia", evidence=[
        {"field": "funder", "page": 9, "basis": "inferred",
         "quote": "Cotton Australia: the industry's peak body, funded by voluntary grower levies"}])
    stated = DocumentMetadata(funder="CRDC", evidence=[
        {"field": "funder", "page": 2, "quote": "This project was funded by CRDC", "basis": "stated"}])
    assert tidy(figure).funder is None
    assert tidy(guessed).funder is None
    assert tidy(stated).funder == "Cotton Research and Development Corporation"


def test_a_corporate_author_named_in_the_organisation_quote_is_the_organisation() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    quote = "This Sustainability Report has been developed by the Sustainability Working Group on behalf of the industry."
    metadata = tidy(DocumentMetadata(
        authors=["Sustainability Working Group (on behalf of the Australian cotton industry)"],
        organisation="Cotton Australia",
        evidence=[{"field": "organisation", "page": 37, "quote": quote, "basis": "stated"}]))
    assert metadata.authors == ["Sustainability Working Group"]
    assert metadata.organisation == "Sustainability Working Group"


def test_a_publisher_among_joint_copyright_holders_publishes_with_them() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(publisher="Cotton Australia", copyright_holder="CRDC; Cotton Australia"))
    assert metadata.publisher == "Cotton Research and Development Corporation; Cotton Australia"


def test_a_later_copyright_year_sets_the_publication_year() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    pages = ["Australian Cotton Sustainability Report 2024", "© CRDC 2025. © Cotton Australia 2025."]
    assert tidy(DocumentMetadata(publication_date="2024"), pages).publication_date == "2025"
    assert tidy(DocumentMetadata(publication_date="2018-11-22"), ["© 2018 Elsevier Ltd."]).publication_date == "2018-11-22"


def test_a_survey_in_the_title_makes_a_survey_report() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    title = "Cotton Growing Practices 2016: Findings of the CRDC's survey of cotton growers"
    assert tidy(DocumentMetadata(title=title, document_type="report")).document_type == "survey report"


def test_organisation_names_damaged_by_ocr_take_the_known_name() -> None:
    from crdc_pipeline.extraction.metadata.names import canonical_names

    assert canonical_names("Australian Cotton Q--owers Research Association Incorporated") == (
        "Australian Cotton Growers Research Association")
    assert canonical_names("Australian Cotton Growers Association") == "Australian Cotton Growers Association"
    assert canonical_names("Australia's cotton growers, Australian Government") == (
        "Australia's cotton growers; Australian Government")


def test_bodies_that_only_received_funding_are_not_funders() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    received = {"field": "funder", "page": 3, "basis": "stated",
                "quote": "Cotton Australia and ACSA received ATMAC funding from the Australian Government"}
    supported = {"field": "funder", "page": 3, "basis": "stated",
                 "quote": "funded by this grant, with additional support from Cotton Australia, ACSA and CRDC"}
    only = tidy(DocumentMetadata(funder="Australian Government; Cotton Australia; ACSA", evidence=[received]))
    both = tidy(DocumentMetadata(funder="Australian Government", evidence=[received, supported]))
    assert only.funder == "Australian Government"
    assert set(both.funder.split("; ")) == {"Australian Government", "Cotton Australia",
                                            "Australian Cotton Shippers Association",
                                            "Cotton Research and Development Corporation"}


def test_a_corporate_author_credited_with_the_work_is_the_organisation() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(
        authors=["Sustainability Working Group"], organisation="Cotton Australia",
        evidence=[{"field": "authors", "page": 37, "basis": "stated",
                   "quote": "This Sustainability Report has been developed by the Sustainability Working Group"},
                  {"field": "organisation", "page": 9, "basis": "stated",
                   "quote": "Cotton Australia: the industry's peak body"}]))
    assert metadata.organisation == "Sustainability Working Group"


def test_collaborating_bodies_in_the_organisation_quote_are_all_kept() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(organisation="Cotton Australia", evidence=[
        {"field": "organisation", "page": 2, "basis": "stated",
         "quote": "Cotton Australia, ACSA and CRDC are collaborating on the roadmap"}]))
    assert metadata.organisation == (
        "Cotton Australia; Australian Cotton Shippers Association; Cotton Research and Development Corporation")


def test_a_copyright_line_is_not_a_funding_statement() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(funder="CRDC", evidence=[
        {"field": "funder", "page": 37, "quote": "© CRDC 2025.", "basis": "stated"}]))
    assert metadata.funder is None


def test_copyright_holders_need_a_printed_copyright_line() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    assert tidy(DocumentMetadata(copyright_holder="AB Hearn"), ["Prepared by AB Hearn"]).copyright_holder is None
    filled = tidy(DocumentMetadata(), ["Annual Report", "© Cotton Research and Development Corporation 2024"])
    assert filled.copyright_holder == "Cotton Research and Development Corporation"


def test_the_citation_credits_a_corporate_author_over_a_contact_person() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    pages = ["Community Trust", "Citation Voconiq (2023) Community Trust in the Australian Cotton Industry. Voconiq, Australia."]
    assert tidy(DocumentMetadata(authors=["Kieren Moffat"]), pages).authors == ["Voconiq"]


def test_a_bodys_own_report_without_named_authors_is_credited_to_it() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    metadata = tidy(DocumentMetadata(organisation="CRDC", publisher="CRDC"))
    assert metadata.authors == ["Cotton Research and Development Corporation"]


def test_a_body_that_only_managed_the_project_is_not_an_author() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    managed = tidy(DocumentMetadata(authors=["Aither"], evidence=[
        {"field": "authors", "page": 10, "quote": "This project is managed by Aither, www.aither.com.au.", "basis": "stated"}]))
    prepared = tidy(DocumentMetadata(authors=["Aither"], evidence=[
        {"field": "authors", "page": 1, "quote": "Prepared by Aither", "basis": "stated"}]))
    assert managed.authors == []
    assert prepared.authors == ["Aither"]


def test_journal_articles_list_every_affiliated_institution() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    page = ("ABSTRACT Results. M.T. Plumblee*, Clemson University, Department of Plant Sciences, Blackville; "
            "D.M. Dodds, Mississippi State University, Department of Plant and Soil Sciences, Starkville; "
            "L.J. Krutz, Mississippi State University, Mississippi Water Resources Research Institute; and "
            "J.N. Jenkins, United States Department of Agriculture, Agricultural Research Service, Starkville. "
            "*Corresponding author: mplumbl@clemson.edu")
    metadata = tidy(DocumentMetadata(
        organisation="Clemson University", journal_article={"journal": "The Journal of Cotton Science"},
        evidence=[{"field": "organisation", "page": 1, "basis": "stated",
                   "quote": "M.T. Plumblee*, Clemson University, Department of Plant Sciences, Blackville"}]), [page])
    assert metadata.organisation == (
        "Clemson University; Mississippi State University; United States Department of Agriculture")


def test_a_journal_is_not_its_own_publisher_and_its_copyright_line_is_not_funding() -> None:
    from crdc_pipeline.extraction.metadata.extract import tidy

    header = "The Journal of Cotton Science 25:1-6 (2021) http://journal.cotton.org, © The Cotton Foundation 2021"
    metadata = tidy(DocumentMetadata(
        publisher="The Journal of Cotton Science", copyright_holder="The Cotton Foundation", funder="The Cotton Foundation",
        evidence=[{"field": "funder", "page": 1, "basis": "stated", "quote": "© The Cotton Foundation 2021"}]), [header])
    assert metadata.publisher == "The Cotton Foundation"
    assert metadata.funder is None
    assert (metadata.journal_article.volume, metadata.journal_article.pages, metadata.journal_article.issue_year) == (
        "25", "1-6", "2021")
