"""Splitting the study's own conditions by dataset, for documents that report more than one dataset.

A report on deaths from coroners' records and on workers compensation claims has two sources, two
collection methods and two sets of limitations; each statistic takes only the facts of its own dataset
plus those shared by all."""

from __future__ import annotations

import re

from ..backends import Backend
from ..statistics.schema import Statistic
from .prompts import (
    ASSIGN_PROMPT,
    DATASET_PROMPT,
    METHOD_KINDS,
    METHOD_REACH_PROMPT,
    DatasetAssignments,
    MethodReaches,
    StudyDatasets,
)
from .schema import ContextItem


# Kinds that describe how data were analysed rather than which data: a group of these alone is not a dataset.
ANALYSIS_KINDS = {"study_design", "preprocessing"}


def split_by_dataset(
    study: list[str], items: list[ContextItem], statistics: list[Statistic], generate
) -> tuple[dict[str, list[str]], list[str], dict[str, int]]:
    """(each dataset's item IDs by name, shared item IDs, each statistic's dataset index).

    With one dataset or none, everything is shared and nothing is assigned.
    """
    by_id = {i.item_id: i for i in items}
    listed = "\n".join(f"{i} | {by_id[i].item_type} | {by_id[i].value} | {by_id[i].quote}" for i in study)
    answer = generate(DATASET_PROMPT.format(items=listed), StudyDatasets, "datasets")
    known = set(study)
    groups, seen = [], set()
    for dataset in answer.datasets:
        ids = [i for i in dict.fromkeys(dataset.item_ids) if i in known and i not in seen]
        # A group of method facts alone is an analysis of the data ("path modelling", "mean scores"), not a
        # dataset of its own, and its facts stay shared.
        if any(by_id[i].item_type not in ANALYSIS_KINDS for i in ids):
            groups.append((dataset.name, ids))
            seen.update(ids)
    if len(groups) < 2:
        return {}, study, {}
    shared = [i for i in study if i not in seen]
    datasets = "\n".join(f"- {name}: " + "; ".join(by_id[i].value for i in ids) for name, ids in groups)
    names = [name.lower() for name, _ in groups]
    assigned: dict[str, int] = {}
    for batch in _batches([s for s in statistics if s.record_type != "cited"]):
        picked = generate(ASSIGN_PROMPT.format(datasets=datasets, statistics=_rows(batch)), DatasetAssignments,
                          "dataset assignment")
        for n, entry in enumerate(picked.datasets):
            # By name; an answer whose name was reworded is taken by its position in the list.
            index = names.index(entry.dataset.lower()) if entry.dataset.lower() in names else n
            if index < len(groups):
                for listed_id in entry.statistic_ids:
                    assigned.setdefault(_listed_id(listed_id), index)
    return dict(groups), shared, assigned


# Numbers per call: a few hundred in one prompt made the model reason until its context ran out.
STATISTICS_PER_CALL = 120


def _batches(statistics: list[Statistic]) -> list[list[Statistic]]:
    return [statistics[k:k + STATISTICS_PER_CALL] for k in range(0, len(statistics), STATISTICS_PER_CALL)]


def _rows(statistics: list[Statistic]) -> str:
    return "\n".join(f"{s.statistic_id} | {s.page} | {s.variable} | {s.reported_text} | {s.quote}" for s in statistics)


def _listed_id(listed: str) -> str:
    """The model sometimes copies the whole listed line ("p5_s01 | 5 | Deaths | six | ...")."""
    return re.split(r"[\s|:]", listed.strip(), maxsplit=1)[0]


def with_study_items(statistic: Statistic, study: list[str], datasets: dict[str, list[str]], shared: list[str],
                     assigned: dict[str, int]) -> Statistic:
    """The study's items a statistic takes: all of them with one dataset; otherwise the shared ones and its own
    dataset's, and none of another dataset's (even if the linking step chose them)."""
    if statistic.record_type == "cited":
        return statistic
    groups = list(datasets.values())
    if not groups:
        own = study
        others: set[str] = set()
    else:
        index = assigned.get(statistic.statistic_id, -1)
        own = shared + (groups[index] if index >= 0 else [])
        others = {i for n, ids in enumerate(groups) if n != index for i in ids}
    links = [i for i in dict.fromkeys(own + statistic.context_item_ids) if i not in others or i in own]
    return statistic.model_copy(update={"context_item_ids": links})


WIDE_REACH = 1 / 3


def method_reach(
    study: list[str], items: list[ContextItem], statistics: list[Statistic], generate
) -> dict[str, set[str]]:
    """The statistics each of the study's method facts applies to; facts that apply to all are left out.

    The scope step picks method facts that the text applies to the whole study, but a small model also picks
    facts about one measure or one analysis (path modelling, mean scores), which then reach every number.
    """
    by_id = {i.item_id: i for i in items}
    methods = [i for i in study if by_id[i].item_type in METHOD_KINDS]
    numbers = [s for s in statistics if s.record_type != "cited"]
    if not methods or not numbers:
        return {}
    listed = "\n".join(f"{i} | {by_id[i].item_type} | {by_id[i].value} | {by_id[i].quote}" for i in methods)
    covered: dict[str, set[str]] = {i: set() for i in methods}
    for batch in _batches(numbers):
        answer = generate(METHOD_REACH_PROMPT.format(items=listed, statistics=_rows(batch)), MethodReaches, "method reach")
        ids = {s.statistic_id for s in batch}
        narrowed = {f.item_id: {_listed_id(x) for x in f.statistic_ids} & ids
                    for f in answer.facts if f.item_id in covered and not f.applies_to_all}
        for item_id in covered:  # a fact the model left out, or applied to all, covers the whole batch
            covered[item_id] |= narrowed.get(item_id, ids)
    # A fact kept on a large share of the numbers is the study's own; the model only listed some of them.
    return {i: ids for i, ids in covered.items() if len(ids) < WIDE_REACH * len(numbers)}
