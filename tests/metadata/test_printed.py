"""Fields read from fixed printed formats rather than by a model."""

from crdc_pipeline.extraction.metadata.extract import DocumentMetadata, tidy
from crdc_pipeline.extraction.metadata.printed import iso_date, printed_fields

HISTORY_PAGE = (
    "Journal of Cleaner Production 212 (2019) 974-985\n0959-6526/ 2018 Elsevier Ltd.\n"
    "Article history: Received 19 May 2018 reduction are increasing\n"
    "Received in revised form the high yields associated with cotton\n10 October 2018\n"
    "Accepted 21 November 2018 contribution of\nAvailable online 22 November 2018\n"
    "https://doi.org/10.1016/j.jclepro.2018.11.190"
)


def test_reads_article_history_and_doi() -> None:
    assert printed_fields([HISTORY_PAGE]) == {
        "revised_date": "2018-10-10",
        "submission_date": "2018-05-19",
        "acceptance_date": "2018-11-21",
        "publication_date": "2018-11-22",
        "doi": "10.1016/j.jclepro.2018.11.190",
        "journal": "Journal of Cleaner Production",
        "volume": "212",
        "issue_year": "2019",
        "pages": "974-985",
        "issn": "0959-6526",
    }


def test_ignores_a_lone_received_date_and_dois_after_page_two() -> None:
    pages = ["We received 3 May 2020 survey responses.", "body", "https://doi.org/10.1/cited.paper"]
    assert printed_fields(pages) == {}


def test_printed_facts_override_the_model_and_dates_become_iso() -> None:
    model_answer = DocumentMetadata(
        journal_article={"submission_date": "21 November 2018", "acceptance_date": "19 May 2018"}
    )
    fixed = tidy(model_answer, [HISTORY_PAGE]).journal_article
    assert (fixed.submission_date, fixed.acceptance_date) == ("2018-05-19", "2018-11-21")
    assert tidy(DocumentMetadata(), [HISTORY_PAGE]).journal_article.journal == "Journal of Cleaner Production"
    assert tidy(DocumentMetadata(), ["A report with no article history."]).journal_article is None
    assert tidy(DocumentMetadata(publication_date="November 13, 2019")).publication_date == "2019-11-13"


def test_iso_date_keeps_the_written_precision() -> None:
    assert iso_date("December 2025") == "2025-12"
    assert iso_date("2019") == "2019"
    assert iso_date("not a date") == "not a date"


def test_issn_check_digit_rejects_look_alikes() -> None:
    from crdc_pipeline.extraction.metadata.printed import valid_issn

    assert valid_issn("0959-6526")
    assert not valid_issn("2011-2012")


def test_citation_year_comes_from_the_suggested_citation() -> None:
    from crdc_pipeline.extraction.metadata.printed import citation_year

    assert citation_year(["Cover", "Title", "Citation Roth Rural (2017) Cotton Growing Practices 2016"]) == "2017"
    assert citation_year(["CITATION Voconiq (2023) Community Trust in the"]) == "2023"
    assert citation_year(["Received 19 May 2018 (Smith, 2015)"]) is None
