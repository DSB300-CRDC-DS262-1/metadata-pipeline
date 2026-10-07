from crdc_pipeline.extraction.metadata.scoring import field_value, matches, same


def test_same_ignores_case_spacing_and_order() -> None:
    assert same(" A Title ", "a title")
    assert same(["B", "A"], ["a", "b"])
    assert same("NSW DPI; CRDC", "crdc; nsw dpi")
    assert not same(None, "x")
    assert same(None, None)


def test_field_value_reads_dotted_names_and_missing_blocks() -> None:
    article = {"journal_article": {"volume": "212"}}
    assert field_value(article, "journal_article.volume") == "212"
    assert field_value({"journal_article": None}, "journal_article.volume") is None


def test_organisation_fields_compare_standard_names() -> None:
    assert matches("funder", "Australian Cotton Research and Development Corporation (CRDC)",
                   "Cotton Research and Development Corporation")
    assert not matches("title", "Australian Cotton Research and Development Corporation (CRDC)",
                       "Cotton Research and Development Corporation")


def test_sub_units_count_as_their_organisation() -> None:
    assert matches("organisation",
                   "NSW DPI; Faculty of Science, Sydney Institute of Agriculture, The University of Sydney",
                   "NSW Department of Primary Industries; The University of Sydney")


def test_authors_compare_without_titles() -> None:
    assert matches("authors", ["Tony Lower"], ["Hon. A/Prof Tony Lower"])
    assert not matches("authors", ["Tony Lower"], ["Kerri-Lynn Peachey"])
