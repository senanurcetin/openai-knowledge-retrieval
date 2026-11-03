"""Regression tests for deterministic, duplicate-free RAG dataset sampling."""
from __future__ import annotations

from evals.generator.sampler import sample_by_mix


def _candidate(label, difficulty="easy", tags=None, source=None):
    return {
        "question": label,
        "correct_answer": f"answer-{label}",
        "difficulty": difficulty,
        "tags": tags or [],
        "source_id": source,
    }


def test_fill_phase_never_selects_a_candidate_twice():
    rows = [
        _candidate("easy-1", "easy"),
        _candidate("easy-2", "easy"),
        _candidate("medium-1", "medium"),
        _candidate("medium-2", "medium"),
    ]
    chosen = sample_by_mix(rows, mix={"easy": 0.5, "medium": 0.5}, total=6)
    assert len(chosen) == 4
    assert {x["question"] for x in chosen} == {
        "easy-1", "easy-2", "medium-1", "medium-2",
    }
    assert len({id(x) for x in chosen}) == len(chosen)


def test_source_and_tag_caps_apply_across_both_sampling_phases():
    rows = [
        _candidate("a1", "easy", tags=["tag-a"], source="source-a"),
        _candidate("a2", "easy", tags=["tag-a"], source="source-a"),
        _candidate("b1", "medium", tags=["tag-b"], source="source-b"),
        _candidate("b2", "medium", tags=["tag-b"], source="source-b"),
        _candidate("c1", "hard", tags=["tag-c"], source="source-c"),
    ]
    chosen = sample_by_mix(
        rows, mix={"easy": 0.5, "medium": 0.25, "hard": 0.25},
        per_tag_cap=1, per_source_cap=1, total=8,
    )
    assert {row["question"] for row in chosen} == {"a1", "b1", "c1"}
    assert len(chosen) == 3


def test_repeatable_sampling_uses_stable_difficulty_order():
    rows = [
        _candidate("hard", "hard"),
        _candidate("medium", "medium"),
        _candidate("easy", "easy"),
    ]
    kwargs = dict(mix={"easy": 0.34, "medium": 0.34, "hard": 0.34}, total=3)
    first = [row["question"] for row in sample_by_mix(rows, **kwargs)]
    assert first == ["easy", "medium", "hard"]
    assert all(
        [row["question"] for row in sample_by_mix(rows, **kwargs)] == first
        for _ in range(20)
    )


def test_missing_difficulty_counts_as_easy_in_target_phase():
    rows = [
        _candidate("default-easy", difficulty=None),
        _candidate("explicit-easy", difficulty="easy"),
        _candidate("medium", difficulty="medium"),
    ]
    chosen = sample_by_mix(rows, mix={"easy": 0.5, "medium": 0.5}, total=4)
    assert len(chosen) == 3
    assert {row["question"] for row in chosen} == {
        "default-easy", "explicit-easy", "medium",
    }


def test_sampling_does_not_modify_input_records():
    rows = [_candidate("one", tags=["tag-a"]), _candidate("two", tags=["tag-b"])]
    snapshot = [row.copy() for row in rows]
    sample_by_mix(rows, mix={"easy": 1.0}, total=4)
    assert rows == snapshot



def test_missing_source_ids_respect_same_source_cap_across_both_phases():
    rows = [
        _candidate("unlabeled-first", "easy", source=None),
        _candidate("unlabeled-second", "medium", source=None),
        _candidate("unlabeled-third", "hard", source=""),
        _candidate("labeled", "medium", source="doc-1"),
    ]
    chosen = sample_by_mix(
        rows, mix={"easy": 0.25, "medium": 0.5, "hard": 0.25},
        total=8, per_source_cap=1,
    )
    assert [row["question"] for row in chosen] == [
        "unlabeled-first", "labeled",
    ]


def test_duplicate_tags_within_one_candidate_consume_one_tag_slot():
    rows = [
        _candidate("duplicate-labels", tags=["safety", "safety"], source="doc-1"),
        _candidate("next-candidate", tags=["safety"], source="doc-2"),
        _candidate("third-candidate", tags=["safety"], source="doc-3"),
    ]
    chosen = sample_by_mix(
        rows, mix={"easy": 1.0}, total=3, per_tag_cap=2,
    )
    assert [row["question"] for row in chosen] == [
        "duplicate-labels", "next-candidate",
    ]


def test_empty_source_and_duplicate_tags_do_not_mutate_inputs():
    rows = [
        _candidate("unlabeled", tags=["a", "a"], source=None),
        _candidate("another", tags=["a"], source=None),
    ]
    before = [(row.copy(), list(row["tags"])) for row in rows]
    result = sample_by_mix(
        rows, mix={"easy": 1.0}, total=3,
        per_source_cap=1, per_tag_cap=2,
    )
    assert len(result) == 1
    assert [(row, row["tags"]) for row in rows] == before
