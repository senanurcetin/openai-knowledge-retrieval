from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def sample_by_mix(
    items: List[Dict[str, Any]],
    mix: Dict[str, float],
    per_tag_cap: int = 30,
    total: int = 150,
    per_source_cap: int | None = None,
) -> List[Dict[str, Any]]:
    """Sample unique input candidates by difficulty, enforcing caps in both passes.

    The target pass uses a stable easy/medium/hard order. The fill pass
    considers remaining original input slots only; it never selects an
    already sampled candidate again merely to reach the requested total.
    """
    difficulties = ("easy", "medium", "hard")
    buckets: Dict[str, List[int]] = defaultdict(list)
    normalized_difficulty: list[str] = []
    for idx, candidate in enumerate(items):
        difficulty = (candidate.get("difficulty") or "easy").lower()
        normalized_difficulty.append(difficulty)
        buckets[difficulty].append(idx)

    out: List[Dict[str, Any]] = []
    selected_indexes: set[int] = set()
    tag_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    difficulty_counts: Dict[str, int] = defaultdict(int)

    def select(index: int) -> bool:
        if index in selected_indexes or len(out) >= total:
            return False
        candidate = items[index]
        tags = candidate.get("tags") or []
        source = candidate.get("source_id") or ""
        if any(tag_counts[tag] >= per_tag_cap for tag in tags):
            return False
        if per_source_cap is not None and source_counts[source] >= per_source_cap:
            return False

        selected_indexes.add(index)
        out.append(candidate)
        for tag in tags:
            tag_counts[tag] += 1
        if source:
            source_counts[source] += 1
        difficulty_counts[normalized_difficulty[index]] += 1
        return True

    targets = {diff: int(total * mix.get(diff, 0.0)) for diff in difficulties}
    for difficulty in difficulties:
        for index in buckets.get(difficulty, []):
            if len(out) >= total or difficulty_counts[difficulty] >= targets[difficulty]:
                break
            select(index)

    # Fill unallocated slots without replaying candidates from the first pass.
    for index in range(len(items)):
        if len(out) >= total:
            break
        select(index)
    return out
