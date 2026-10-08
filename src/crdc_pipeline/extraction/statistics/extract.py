"""Statistics and context extraction, page by page.

Pass 1 lists each page's context items (how numbers were produced and should be read). Pass 2 lists each
page's statistics, and pass 3 links them to those items by ID. Pages are processed one at a time so a
small local model never needs the whole document in its context window. IDs, page numbers and contexts
are set in code. The context steps' prompts and rules are in extraction/contexts/."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..backends import Backend
from ..contexts.datasets import method_reach, split_by_dataset, with_study_items
from ..contexts.links import (
    document_link_prompt,
    fit_questions,
    fit_study_links,
    item_list,
    questions_from_quotes,
    trim_links,
)
from ..contexts.prompts import (
    CONTEXT_PROMPT,
    DOCUMENT_CONTEXT_PROMPT,
    LINK_PROMPT,
    MAX_METHOD_ITEMS,
    METHOD_KINDS,
    METHOD_SCOPE_PROMPT,
    SCOPE_PROMPT,
    DocumentContextItems,
    DocumentLinks,
    PageContextItems,
    PageLinks,
    StudyItems,
)
from ..contexts.schema import STATISTIC_LEVEL, Context, ContextItem, group_contexts
from ..contexts.scope import KEY_KINDS, is_label, not_a_condition, running_lines, sharpen_scope
from .duplicates import (
    drop_conversions,
    drop_duplicates,
    duplicate_candidates,
    duplicate_prompt,
    merge_ranges,
    pair_numbers,
)
from .filters import (
    CAPTION,
    axis_ticks,
    before_references,
    is_structural,
    is_table_row,
    value_not_in_quote,
    value_not_on_page,
    without_tables,
    word_number_note,
)
from .periods import date_survey_answers, survey_year, with_item_period, with_period
from .prompts import STATISTICS_PROMPT, PageStatistics, SamePairs
from .records import answers_to_questions, modelled_results
from .schema import Statistic
from .values import label_change, label_cited, label_survey_answer, label_type, shared_unit, tidy_statistic


DUPLICATE_BATCH = 50


class FindingsExtractionResult(BaseModel):
    context_items: list[ContextItem]
    contexts: list[Context]
    statistics: list[Statistic]
    context_ids: dict[str, str | None]
    resolved_model: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    failed_steps: list[str] = Field(default_factory=list)
    study_item_ids: list[str] = Field(default_factory=list)
    datasets: dict[str, list[str]] = Field(default_factory=dict)
    dataset_of: dict[str, str] = Field(default_factory=dict)
    method_reach: dict[str, list[str]] = Field(default_factory=dict)


def extract_findings(
    pages: list[str],
    backend: Backend,
    document_id: str,
    exclude_tables: bool = False,
    link: bool = True,
    scope: bool = True,
    context_backend: Backend | None = None,
    whole_document: bool = False,
    dedupe: bool = True,
    datasets: bool = True,
    drop_table_rows: bool = False,
) -> FindingsExtractionResult:
    """Context items, then statistics, for every page with text.

    `scope` asks once which items describe the document's own study as a whole and links them to
    every statistic that is not cited from elsewhere; `link` adds a linking pass per page.
    `context_backend` runs the context, scope and linking steps on a different model; each step
    runs over all pages before the next, so two local models are swapped as rarely as possible.
    `whole_document` runs the context and linking steps once over the whole document instead of
    page by page (statistics stay page by page). `datasets` splits the study's items by dataset when the
    document reports more than one, so each statistic takes only its own dataset's conditions.
    """
    pages = before_references(pages)
    if exclude_tables:
        pages = [without_tables(p) for p in pages]
    context_backend = context_backend or backend
    usage = {"seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0}
    models: list[str] = []
    failed: list[str] = []

    def generate(prompt: str, schema: type[BaseModel], step: str, model: Backend) -> BaseModel:
        try:
            result = model.generate(prompt, schema)
        except RuntimeError as exc:  # e.g. the model looping until the server stops it
            failed.append(f"{step}: {exc}")
            return schema()
        usage["seconds"] += result.seconds
        usage["prompt_tokens"] += result.prompt_tokens or 0
        usage["output_tokens"] += result.output_tokens or 0
        if result.model not in models:
            models.append(result.model)
        return result.parsed

    numbered = [(n, text) for n, text in enumerate(pages, start=1) if text.strip()]
    items: list[ContextItem] = []
    if whole_document:
        document = "\n\n".join(f"=== page {n} ===\n{text}" for n, text in numbered)
        found = generate(DOCUMENT_CONTEXT_PROMPT.format(text=document), DocumentContextItems, "document context", context_backend).items
        pages_with_text = {n for n, _ in numbered}
        for k, item in enumerate((i for i in found if i.page in pages_with_text), start=1):
            items.append(item.model_copy(update={"item_id": f"p{item.page}_i{k:03d}"}))
    else:
        for n, text in numbered:
            prompt = CONTEXT_PROMPT.format(page=n, text=text)
            for k, item in enumerate(generate(prompt, PageContextItems, f"page {n} context", context_backend).items, start=1):
                items.append(item.model_copy(update={"item_id": f"p{n}_i{k:02d}", "page": n}))
    # A figure or table caption only says what a chart shows; it never matched a real context fact. Running
    # headers and footers, thanks and disclaimers, and generic words ("Survey") are not conditions either.
    running = running_lines(pages)
    items = [with_item_period(i) for i in items if not CAPTION.match(i.quote) and not is_label(i, pages[i.page - 1])
             and not not_a_condition(i, running)]

    known = {i.item_id for i in items}
    study: list[str] = []
    if scope and items:
        chosen = generate(SCOPE_PROMPT.format(items=item_list(items)), StudyItems, "study scope", context_backend).item_ids
        per_statistic = {i.item_id for i in items if i.item_type in STATISTIC_LEVEL}
        study = sharpen_scope([i for i in dict.fromkeys(chosen) if i in known and i not in per_statistic], items)
        # The first call favours the key kinds; a second, narrower question finds whole-study method facts.
        methods = [i for i in items if i.item_type in METHOD_KINDS]
        if methods:
            picked = generate(METHOD_SCOPE_PROMPT.format(items=item_list(methods)), StudyItems, "method scope",
                              context_backend).item_ids
            method_ids = {i.item_id for i in methods}
            study += [i for i in dict.fromkeys(picked) if i in method_ids and i not in study][:MAX_METHOD_ITEMS]

    by_page: list[tuple[int, list[Statistic]]] = []
    for n, text in numbered:
        prompt = STATISTICS_PROMPT.format(page=n, text=text, items=item_list(items), word_numbers=word_number_note(text))
        found = [s for s in generate(prompt, PageStatistics, f"page {n} statistics", backend).statistics
                 if not (drop_table_rows and is_table_row(s, text)) and not is_structural(s) and not value_not_in_quote(s)
                 and not value_not_on_page(s, text)]
        ticks = {id(s) for s in axis_ticks(found)}
        found = [s for s in found if id(s) not in ticks]
        kept = merge_ranges(drop_conversions([with_period(label_type(shared_unit(label_change(label_cited(label_survey_answer(s, text))))))
                                 for s in (tidy_statistic(s) for s in found) if s is not None]))
        by_page.append((n, [
            statistic.model_copy(update={
                "statistic_id": f"p{n}_s{k:02d}",
                "page": n,
                "context_item_ids": [i for i in statistic.context_item_ids if i in known],
            })
            for k, statistic in enumerate(kept, start=1)
        ]))

    if link and whole_document and any(stats for _, stats in by_page):
        prompt = document_link_prompt(items, [s for _, stats in by_page for s in stats])
        links = generate(prompt, DocumentLinks, "document links", context_backend).links
        extra = {entry.statistic_id: [i for i in entry.item_ids if i in known] for entry in links}
        by_page = [
            (n, [s.model_copy(update={"context_item_ids": list(dict.fromkeys(s.context_item_ids + extra.get(s.statistic_id, [])))})
                 for s in stats])
            for n, stats in by_page
        ]

    statistics: list[Statistic] = []
    for n, page_statistics in by_page:
        if link and not whole_document and page_statistics:
            listed = "\n".join(
                f"{s.statistic_id} | {s.variable} | {s.reported_text} | {s.quote}" for s in page_statistics
            )
            prompt = LINK_PROMPT.format(page=n, items=item_list(items), statistics=listed)
            links = generate(prompt, PageLinks, f"page {n} links", context_backend).links
            extra = {entry.statistic_id: [i for i in entry.item_ids if i in known] for entry in links}
            page_statistics = [
                s.model_copy(update={"context_item_ids": list(dict.fromkeys(s.context_item_ids + extra.get(s.statistic_id, [])))})
                for s in page_statistics
            ]
        statistics.extend(page_statistics)

    # The study's own conditions apply to all of its numbers (or, with several datasets, to the numbers of
    # their own dataset), but not to figures cited from other studies.
    groups, shared, dataset_index, reach = {}, study, {}, {}
    if datasets:
        # Items the context step marked as applying to the whole document join the study's items, so they
        # are split by dataset too, instead of reaching every number (cover-page dates, one dataset's limits).
        # A key kind the study already states (its period, region, sample, collection) is not added again: the
        # scope step chose the method's own statement over vaguer ones ("national survey"). Data sources are,
        # as a study with several datasets has several.
        by_id = {i.item_id: i for i in items}
        settled = {by_id[i].item_type for i in study} & (set(KEY_KINDS) - {"data_source"})
        study = study + [i.item_id for i in items if i.whole_document and i.item_type not in STATISTIC_LEVEL
                         and i.item_id not in study and i.item_type not in settled]
        items = [i.model_copy(update={"whole_document": False}) if i.whole_document else i for i in items]
        shared = study
    if datasets and len(study) > 1 and statistics:
        groups, shared, dataset_index = split_by_dataset(study, items, statistics,
                                                    lambda p, schema, step: generate(p, schema, step, context_backend))
    statistics = [with_study_items(s, study, groups, shared, dataset_index) for s in statistics]
    if datasets and study and statistics:
        # Method facts that cover only some numbers (one measure, one analysis) come off the rest.
        reach = method_reach(study, items, statistics, lambda p, schema, step: generate(p, schema, step, context_backend))
        statistics = [s.model_copy(update={"context_item_ids": [i for i in s.context_item_ids
                                                                 if i not in reach or s.statistic_id in reach[i]]})
                      for s in statistics]

    if dedupe:
        pairs = duplicate_candidates(statistics)
        # A few dozen pairs per call: a long list made the model reason until its context ran out.
        same: set[int] = set()
        for start in range(0, len(pairs), DUPLICATE_BATCH):
            batch = pairs[start:start + DUPLICATE_BATCH]
            chosen = generate(duplicate_prompt(batch), SamePairs, "duplicates", backend).pair_ids
            same |= {start + k for k in pair_numbers(chosen, len(batch))}
        if same:
            statistics = drop_duplicates(statistics, same, pairs)
    statistics = trim_links(statistics, items, study)
    statistics = modelled_results(answers_to_questions(statistics, items), items)
    statistics = fit_questions(fit_study_links(statistics, items, study), items)
    statistics, items = questions_from_quotes(statistics, items)
    statistics = date_survey_answers(statistics, survey_year(items, study))
    contexts, assigned = group_contexts(document_id, items, statistics)
    return FindingsExtractionResult(
        context_items=items,
        contexts=contexts,
        statistics=statistics,
        context_ids=assigned,
        resolved_model=" + ".join(models),
        seconds=usage["seconds"],
        prompt_tokens=usage["prompt_tokens"],
        output_tokens=usage["output_tokens"],
        failed_steps=failed,
        study_item_ids=study,
        datasets=groups,
        dataset_of={k: list(groups)[v] for k, v in dataset_index.items()},
        method_reach={k: sorted(v) for k, v in reach.items()},
    )
