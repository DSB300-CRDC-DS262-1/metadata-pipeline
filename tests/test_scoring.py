from crdc_pipeline.extraction.scoring import dates_equal, field_matches, parse_date, strict_equal


def test_parse_date_handles_common_formats() -> None:
    assert parse_date("2019-11-13") == ((2019, 11, 13), 3)
    assert parse_date("November 13, 2019") == ((2019, 11, 13), 3)
    assert parse_date("21 November 2018") == ((2018, 11, 21), 3)
    assert parse_date("December 2025") == ((2025, 12), 2)
    assert parse_date("2019") == ((2019,), 1)
    assert parse_date("not a date") is None


def test_dates_equal_compares_at_the_coarser_precision() -> None:
    assert dates_equal("November 13, 2019", "2019-11-13")
    assert dates_equal("2025-12", "2025-12-01")
    assert not dates_equal("2019", "2018-11-22")
    assert not dates_equal("2019-11-22", "2018-11-22")
    assert dates_equal(None, None)
    assert not dates_equal(None, "2019")


def test_field_matches_only_normalises_date_fields_when_asked() -> None:
    assert not field_matches("publication_date", "November 13, 2019", "2019-11-13")
    assert field_matches("publication_date", "November 13, 2019", "2019-11-13", normalise_dates=True)
    assert not field_matches("title", "A", "B", normalise_dates=True)


def test_strict_equal_ignores_case_whitespace_and_author_order() -> None:
    assert strict_equal(" A Title ", "a title")
    assert strict_equal(["B", "A"], ["a", "b"])
    assert not strict_equal(None, "x")
