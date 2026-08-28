"""Protocol oracles shared by EvoMemBench runners."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from bench.agent_memory.evomembench.tool_dataset import (
    PINNED_TOOL_ROW_COUNT,
    TOOL_CATEGORIES,
    ToolCorpus,
)


def validate_tool_protocol(
    corpus: ToolCorpus, category_by_id: Mapping[str, str]
) -> dict[str, int]:
    source_ids = {episode.source_id for episode in corpus.episodes}
    partition_ids = set(category_by_id)
    if source_ids != partition_ids:
        raise ValueError(
            "CrossEp-Tool prompt ids and protocol partitions differ: "
            f"missing={sorted(source_ids - partition_ids)[:5]}, "
            f"extra={sorted(partition_ids - source_ids)[:5]}"
        )
    if len(source_ids) != PINNED_TOOL_ROW_COUNT:
        raise ValueError(f"CrossEp-Tool has {len(source_ids)} unique ids, expected 200")
    counts = Counter(category_by_id.values())
    expected = {category: 50 for category in TOOL_CATEGORIES}
    if dict(counts) != expected:
        raise ValueError(f"CrossEp-Tool category counts differ: {dict(counts)}")
    return expected


def ordered_transfer_pairs(
    categories: Sequence[str] = TOOL_CATEGORIES,
) -> tuple[tuple[str, str], ...]:
    if len(set(categories)) != len(categories):
        raise ValueError("transfer categories must be unique")
    return tuple(
        (source, target)
        for source in categories
        for target in categories
        if source != target
    )
