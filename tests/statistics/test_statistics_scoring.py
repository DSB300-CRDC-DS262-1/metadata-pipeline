"""Scoring statistics against the reference."""

from crdc_pipeline.extraction.statistics.scoring import match_statistics, score_document, values_match
from crdc_pipeline.reference import findings_reference


REFERENCE = {"documents": findings_reference()}


def test_reference_scores_perfectly_against_itself() -> None:
    for document in REFERENCE["documents"]:
        score = score_document(document, document)
        assert score["matched"] == score["reference"], document["document_id"]
        assert all(a == n for a, n in score["agreement"].values())
        correct, wanted, given = score["context_links"]
        assert correct == wanted == given
        assert all(wrong == missing == 0 for _, wrong, missing in score["key_conditions"].values())


def test_values_match_within_rounding_and_bounds() -> None:
    assert values_match({"value": 10.28}, {"value": 10.3})
    assert values_match({"value": None, "value_max": 5}, {"value": 5})
    assert values_match({"value": None, "value_min": 3, "value_max": 5}, {"value": None, "value_min": 3, "value_max": 5})
    assert not values_match({"value": 0.07}, {"value": 0.09})
    assert not values_match({"value": 47}, {"value": 0.47})


def test_matching_prefers_the_same_answer_category() -> None:
    reference = [{"variable": "disagree with x", "value": 42, "page": 1, "response_category": "neutral"}]
    extracted = [{"variable": "x", "value": 42, "page": 1, "response_category": "agree"},
                 {"variable": "x", "value": 42, "page": 1, "response_category": "neutral"}]
    assert match_statistics(reference, extracted)[0][1]["response_category"] == "neutral"


def test_values_on_another_page_do_not_match() -> None:
    assert match_statistics([{"variable": "a", "value": 6, "page": 1}], [{"variable": "a", "value": 6, "page": 2}]) == []
