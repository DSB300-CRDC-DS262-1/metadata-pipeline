"""The statistics and context pipeline with a fake model (no model needed)."""

from crdc_pipeline.extraction.backends import GenerationResult
from crdc_pipeline.extraction.contexts.prompts import (
    MethodReaches,
    PageContextItems,
    PageLinks,
    StatisticLinks,
    StudyDatasets,
    StudyItems,
)
from crdc_pipeline.extraction.contexts.schema import ContextItem
from crdc_pipeline.extraction.statistics.extract import extract_findings
from crdc_pipeline.extraction.statistics.prompts import PageStatistics, SamePairs
from crdc_pipeline.extraction.statistics.schema import Statistic


class FakeBackend:
    label = "fake"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, schema, images=None):
        self.prompts.append(prompt)
        if schema is StudyItems:
            parsed = StudyItems()
        elif schema is SamePairs:
            parsed = SamePairs()
        elif schema is StudyDatasets:
            parsed = StudyDatasets()
        elif schema is MethodReaches:
            parsed = MethodReaches()
        elif schema is PageLinks:
            parsed = PageLinks(links=[StatisticLinks(statistic_id="p1_s01", item_ids=["p3_i01", "made_up"])])
        elif schema is PageContextItems:
            parsed = PageContextItems(items=[ContextItem(item_id="item_01", item_type="data_source", value="NCIS",
                                                         quote="Data were drawn from the NCIS database.", page=99)])
        else:
            parsed = PageStatistics(statistics=[Statistic(
                statistic_id="stat_01", variable="deaths", value=6, reported_text="six", unit="deaths",
                statistic_type="count", record_type="measurement", value_status="exact",
                context_item_ids=["p1_i01", "made_up"], page=99, quote="six deaths")])
        return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=10, output_tokens=5)


def test_ids_pages_and_contexts_are_set_in_code() -> None:
    backend = FakeBackend()
    result = extract_findings(["page one text", "", "page three text"], backend, "doc_x", dedupe=False)
    assert [i.item_id for i in result.context_items] == ["p1_i01", "p3_i01"]
    assert [s.statistic_id for s in result.statistics] == ["p1_s01", "p3_s01"]
    assert [s.page for s in result.statistics] == [1, 3]
    assert result.statistics[0].context_item_ids == ["p1_i01", "p3_i01"]  # unknown IDs dropped, links merged
    assert result.statistics[1].context_item_ids == ["p1_i01"]
    assert result.context_ids["p1_s01"] != result.context_ids["p3_s01"]
    assert len(backend.prompts) == 7  # the empty page is skipped
    assert result.seconds == 7.0


def test_statistics_prompt_lists_every_context_item() -> None:
    backend = FakeBackend()
    extract_findings(["first", "second"], backend, "doc_x")
    statistics_prompt = next(p for p in backend.prompts if "List every number" in p)
    assert "=== page 1 ===" in statistics_prompt
    assert "p1_i01 | data_source | NCIS" in statistics_prompt
    assert "p2_i01 | data_source | NCIS" in statistics_prompt


def test_a_failed_page_is_recorded_and_the_rest_continue() -> None:
    class Flaky(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if "=== page 1 ===" in prompt and schema is PageStatistics:
                raise RuntimeError("token repeat limit reached")
            return super().generate(prompt, schema, images)

    result = extract_findings(["one", "two"], Flaky(), "doc_x")
    assert [s.page for s in result.statistics] == [2]
    assert result.failed_steps and "page 1 statistics" in result.failed_steps[0]


def test_linking_pass_adds_items_from_other_pages() -> None:
    class Linking(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema is PageLinks:
                self.prompts.append(prompt)
                ids = ["p2_i01"] if "Numbers on page 1" in prompt else []
                parsed = PageLinks(links=[StatisticLinks(statistic_id="p1_s01", item_ids=ids)])
                return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)
            return super().generate(prompt, schema, images)

    result = extract_findings(["one", "two"], Linking(), "doc_x")
    assert result.statistics[0].context_item_ids == ["p1_i01", "p2_i01"]
    assert extract_findings(["one", "two"], Linking(), "doc_x", link=False).statistics[0].context_item_ids == ["p1_i01"]


def test_whole_study_items_reach_every_statistic_except_cited_ones() -> None:
    class Scoped(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema is StudyItems:
                parsed = StudyItems(item_ids=["p2_i01", "made_up"])
                return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)
            result = super().generate(prompt, schema, images)
            if schema is PageStatistics and "=== page 2 ===" in prompt:
                cited = result.parsed.statistics[0].model_copy(update={"record_type": "cited", "context_item_ids": []})
                result.parsed.statistics = [cited]
            return result

    result = extract_findings(["one", "two"], Scoped(), "doc_x", link=False)
    assert result.study_item_ids == ["p2_i01"]
    assert result.statistics[0].context_item_ids == ["p2_i01", "p1_i01"]
    assert result.statistics[1].context_item_ids == []


def test_context_steps_can_use_another_model() -> None:
    statistics_model, context_model = FakeBackend(), FakeBackend()
    extract_findings(["one", "two"], statistics_model, "doc_x", context_backend=context_model, dedupe=False)
    assert all("List every number" in p for p in statistics_model.prompts)
    assert len(statistics_model.prompts) == 2
    assert not any("List every number" in p for p in context_model.prompts)


def test_whole_document_mode_uses_one_context_call_and_one_link_call() -> None:
    from crdc_pipeline.extraction.contexts.prompts import (
        DocumentContextItems,
        DocumentLinks,
        DocumentStatisticLinks,
    )

    class Whole(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema not in (DocumentContextItems, DocumentLinks):
                return super().generate(prompt, schema, images)
            self.prompts.append(prompt)
            if schema is DocumentContextItems:
                parsed = DocumentContextItems(items=[
                    ContextItem(item_id="x", item_type="limitation", value="provisional", quote="Data were provisional at the time of writing.", page=2),
                    ContextItem(item_id="y", item_type="other", value="no such page", quote="Data were provisional at the time of writing.", page=9)])
            else:
                parsed = DocumentLinks(links=[DocumentStatisticLinks(statistic_id="p1_s01", item_ids=["p2_i001"])])
            return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)

    backend = Whole()
    result = extract_findings(["one", "two"], backend, "doc_x", whole_document=True, scope=False, dedupe=False)
    assert [i.item_id for i in result.context_items] == ["p2_i001"]  # an item on a page with no text is dropped
    assert "=== page 1 ===" in backend.prompts[0] and "=== page 2 ===" in backend.prompts[0]
    assert result.statistics[0].context_item_ids == ["p2_i001"]
    assert len(backend.prompts) == 4  # one context call, two statistics pages, one link call


def test_figure_and_table_captions_are_not_context() -> None:
    class Captions(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema is PageContextItems:
                self.prompts.append(prompt)
                parsed = PageContextItems(items=[
                    ContextItem(item_id="a", item_type="time_period", value="2025", quote="Figure 12: Distribution of response, 2025.", page=1),
                    ContextItem(item_id="b", item_type="data_source", value="NCIS", quote="Data were drawn from the NCIS database.", page=1)])
                return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)
            return super().generate(prompt, schema, images)

    result = extract_findings(["one"], Captions(), "doc_x", link=False, scope=False)
    assert [i.value for i in result.context_items] == ["NCIS"]


def test_a_second_scope_call_adds_whole_study_method_facts() -> None:
    from crdc_pipeline.extraction.contexts.prompts import METHOD_SCOPE_PROMPT

    class Methods(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema is PageContextItems:
                kind = "data_source" if "=== page 1 ===" in prompt else "functional_unit"
                parsed = PageContextItems(items=[ContextItem(item_id="item_01", item_type=kind, value="v", page=1,
                                                             quote="Results are expressed per tonne of lint.")])
                return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)
            if schema is StudyItems:
                method_call = prompt.startswith(METHOD_SCOPE_PROMPT.split("{")[0][:40])
                parsed = StudyItems(item_ids=["p2_i01", "p1_i01"] if method_call else ["p1_i01"])
                return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)
            return super().generate(prompt, schema, images)

    result = extract_findings(["one", "two"], Methods(), "doc_x", link=False)
    assert result.study_item_ids == ["p1_i01", "p2_i01"]  # the method call only adds method kinds


def test_the_pipeline_asks_about_same_value_pairs_and_drops_the_later_one() -> None:
    class Same(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema is SamePairs:
                return GenerationResult(parsed=SamePairs(pair_ids=["pair_001"]), model="fake", seconds=1.0,
                                        prompt_tokens=1, output_tokens=1)
            return super().generate(prompt, schema, images)

    result = extract_findings(["one", "two"], Same(), "doc_x", link=False, scope=False)
    assert [s.page for s in result.statistics] == [1]
    assert len(extract_findings(["one", "two"], Same(), "doc_x", link=False, scope=False, dedupe=False).statistics) == 2


def test_items_marked_whole_document_join_the_study_unless_it_already_states_that_kind() -> None:
    class Marked(FakeBackend):
        def generate(self, prompt, schema, images=None):
            if schema is PageContextItems:
                if "=== page 1 ===" in prompt:
                    found = [ContextItem(item_id="item_01", item_type="collection_method", value="online survey",
                                         quote="An online survey methodology was used.", page=1)]
                else:
                    found = [ContextItem(item_id="item_01", item_type="collection_method", value="national survey",
                                         quote="This national survey asked about trust.", page=2, whole_document=True),
                             ContextItem(item_id="item_02", item_type="limitation", value="self-reported",
                                         quote="All answers were self-reported by participants.", page=2, whole_document=True)]
                return GenerationResult(parsed=PageContextItems(items=found), model="fake", seconds=1.0,
                                        prompt_tokens=1, output_tokens=1)
            if schema is StudyItems:
                return GenerationResult(parsed=StudyItems(item_ids=["p1_i01"]), model="fake", seconds=1.0,
                                        prompt_tokens=1, output_tokens=1)
            return super().generate(prompt, schema, images)

    result = extract_findings(["one", "two"], Marked(), "doc_x", link=False)
    assert result.study_item_ids == ["p1_i01", "p2_i02"]
    assert not any(i.whole_document for i in result.context_items)
