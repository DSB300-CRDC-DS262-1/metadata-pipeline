"""Scoring statistics against the reference."""

from crdc_pipeline.extraction.statistics.scoring import match_statistics, score_document, values_match
from crdc_pipeline.reference import findings_reference


REFERENCE = {"documents": findings_reference()}


def test_reference_scores_perfectly_against_itself() -> None:
    for document in REFERENCE["documents"]:
        score = score_document(document, document)
        assert score["matched"] == score["reference"], document["document_id"]
        assert all(a == n for a, n in score["agreement"].values())
        found, wanted, given, right = score["context_links"]
        assert found == wanted == given == right
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


def test_acceptable_links_count_for_precision_but_not_recall() -> None:
    """Precision by link (needed or acceptable facts); recall by needed fact."""
    from crdc_pipeline.extraction.contexts.scoring import context_links

    reference = {"context_items": [
        {"item_id": "region", "item_type": "geographic_region", "page": 3, "quote": "The study covers northwest New South Wales."},
        {"item_id": "assumed", "item_type": "assumption", "page": 3, "quote": "We assumed that 96% of production was irrigated."}]}
    extracted = {"context_items": [
        {"item_id": "e1", "item_type": "geographic_region", "page": 3, "quote": "The study covers northwest New South Wales."},
        {"item_id": "e2", "item_type": "assumption", "page": 3, "quote": "We assumed that 96% of production was irrigated."}]}
    pairs = [({"context_item_ids": ["region"], "acceptable_item_ids": ["assumed"]}, {"context_item_ids": ["e1", "e2"]})]
    assert context_links(pairs, reference, extracted) == (1, 1, 2, 2)


def test_a_number_spilled_from_a_table_or_chart_is_acceptable_not_required() -> None:
    from crdc_pipeline.extraction.statistics.scoring import score_document

    reference = {"statistics": [], "context_items": [], "contexts": [],
                 "table_values": [{"page": 7, "value": 40.4, "kind": "data"}, {"page": 7, "value": 43.6, "kind": "data"}]}
    extracted = {"context_items": [], "statistics": [
        {"page": 7, "value": 40.4, "variable": "mean claims", "context_item_ids": []},   # a table cell: acceptable
        {"page": 7, "value": 40.4, "variable": "repeat", "context_item_ids": []},        # each table value counts once
        {"page": 8, "value": 43.6, "variable": "other page", "context_item_ids": []},    # another page: not from the table
    ]}
    scores = score_document(reference, extracted)
    assert scores["reference"] == 0 and scores["from_tables"] == 1
