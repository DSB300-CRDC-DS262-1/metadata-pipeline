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


def test_a_reworded_quote_matches_the_reference_fact_sharing_most_of_its_key_words() -> None:
    from crdc_pipeline.extraction.contexts.scoring import item_map

    reference = [
        {"item_id": "i13", "quote": "All figures have been updated to reflect costs in 2018 Australian dollars (previously expressed in 2012 dollars)."},
        {"item_id": "i07", "quote": "We considered economic allocation as the most appropriate means of partitioning."},
        {"item_id": "i01", "quote": "The National Coroners Information System is the sentinel repository of all deaths in Australia."},
    ]
    extracted = [
        {"item_id": "a", "quote": "Earlier data costs (expressed in 2012 dollars), were updated to reflect 2018 Australian dollars."},
        {"item_id": "b", "quote": "The climate change impact on a cradle-to-port basis was 1601 kg CO2e per tonne of lint."},
    ]
    assert item_map(reference, extracted) == {"a": {"i13"}, "b": set()}
