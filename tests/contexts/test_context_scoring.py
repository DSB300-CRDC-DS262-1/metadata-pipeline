"""Scoring context items and links against the reference."""




def test_context_items_match_by_overlapping_quotes() -> None:
    from crdc_pipeline.extraction.contexts.scoring import same_fact

    reference = {"page": 5, "quote": "As with the previous study, the available data for the most recent year (2016-17), is provisional in nature."}
    assert same_fact(reference, {"page": 5, "quote": "the available data for the most recent year (2016-17), is provisional"})
    assert same_fact(reference, {"page": 6, "quote": reference["quote"]})  # the same sentence on another page
    assert not same_fact(reference, {"page": 5, "quote": "Data were accessed through Safe Work Australia."})


def test_key_conditions_separate_wrong_from_missing() -> None:
    from crdc_pipeline.extraction.contexts.scoring import key_conditions

    reference = {"context_items": [
        {"item_id": "ncis", "item_type": "data_source", "page": 3, "quote": "Deaths come from the National Coroners Information System."},
        {"item_id": "swa", "item_type": "data_source", "page": 5, "quote": "Claims data were accessed through Safe Work Australia."}]}
    extracted = {"context_items": [
        {"item_id": "e1", "item_type": "data_source", "page": 3, "quote": "Deaths come from the National Coroners Information System."},
        {"item_id": "e2", "item_type": "data_source", "page": 5, "quote": "Claims data were accessed through Safe Work Australia."}]}
    pairs = [({"context_item_ids": ["ncis"]}, {"context_item_ids": ["e1"]}),
             ({"context_item_ids": ["ncis"]}, {"context_item_ids": ["e2"]}),
             ({"context_item_ids": ["swa"]}, {"context_item_ids": []})]
    assert key_conditions(pairs, reference, extracted)["data_source"] == [1, 1, 1]
