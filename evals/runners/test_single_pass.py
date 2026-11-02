"""Offline integration tests for reusing judged answers and retrieval traces.

Heavy network-backed model, store and grader imports are stubbed; actual
run_local_judge and run_evals functions execute with deterministic fakes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


class FakeItem(SimpleNamespace):
    def model_dump(self):
        return vars(self).copy()


class FakeDecision:
    def __init__(self, decision, correctness, grounding, rationale):
        self.decision = decision
        self.correctness = correctness
        self.grounding = grounding
        self.rationale = rationale

    def model_dump(self):
        return vars(self).copy()


def _module(monkeypatch, name, **attrs):
    mod = ModuleType(name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_common(monkeypatch, rows):
    _module(monkeypatch, "rich", print=lambda *a, **kw: None)
    _module(monkeypatch, "cli.config", RootConfig=object)
    _module(monkeypatch, "evals.datasets.schema", load_and_validate_jsonl=lambda _: rows)


def _cfg(capture=False):
    return SimpleNamespace(
        project="synthetic",
        evals=SimpleNamespace(
            mode="user",
            dataset_path="offline.jsonl",
            judge=SimpleNamespace(model=None),
            thresholds=SimpleNamespace(pass_rate=0.5),
            openai_evals=SimpleNamespace(enabled=False, graders=[]),
            retrieval_metrics=SimpleNamespace(capture=capture, top_k=2),
        ),
    )


def _rows():
    return [
        SimpleNamespace(item=FakeItem(
            id="duplicate-id", question="slow question",
            correct_answer="alpha", citation_text="alpha", source_id="gold"
        )),
        SimpleNamespace(item=FakeItem(
            id="duplicate-id", question="fast question",
            correct_answer="beta", citation_text="beta", source_id=None
        )),
    ]


def test_local_judge_reorders_parallel_completions_and_sends_only_ephemeral_traces(monkeypatch):
    rows = _rows()
    with monkeypatch.context() as ctx:
        _setup_common(ctx, rows)
        _module(ctx, "evals.generator.canonicalize", canonicalize_answer=lambda v: (v or "").lower())
        _module(ctx, "evals.graders.schema", JudgeDecision=FakeDecision)
        _module(ctx, "evals.graders.judge_prompts", SINGLE_RUBRIC_QA_GROUNDED="test")
        _module(
            ctx, "evals.graders.runner", GraderResult=object,
            run_config_graders=lambda *a, **kw: [],
            summarize_graders=lambda *a, **kw: "",
        )
        calls = []

        def fake_answer_query(*, store, cfg, query, **kwargs):
            calls.append((query, kwargs.copy()))
            if query.startswith("slow"):
                time.sleep(0.06)
            answer = "alpha" if query.startswith("slow") else "beta"
            return {"answer_text": answer, "retrieval_trace": {
                "status": "captured",
                "retrieved": [{"doc_id": "gold", "text": answer}],
            }}

        _module(ctx, "retrieval.pipeline", answer_query=fake_answer_query)
        _module(ctx, "stores.base", make_store_from_config=lambda cfg: object())
        judge = _load_source("_test_single_pass_local_judge", "evals/runners/local_judge.py")

        traces = {}
        report = judge.run_local_judge(
            _cfg(capture=True), "offline.jsonl",
            trace_sink=lambda index, trace: traces.__setitem__(index, trace)
        )
        assert [r["question"] for r in report["rows"]] == [
            "slow question", "fast question"
        ]
        assert [r["answer"] for r in report["rows"]] == ["alpha", "beta"]
        assert {q for q, _ in calls} == {"slow question", "fast question"}
        assert len(calls) == 2
        assert all(kwargs == {"capture_retrieval": True} for _, kwargs in calls)
        assert traces[0]["retrieved"][0]["text"] == "alpha"
        assert traces[1]["retrieved"][0]["text"] == "beta"
        assert "retrieval_trace" not in report["rows"][0]
        assert "retrieval_trace" not in report["rows"][1]


def test_local_judge_default_does_not_enable_capture(monkeypatch):
    with monkeypatch.context() as ctx:
        _setup_common(ctx, _rows()[:1])
        _module(ctx, "evals.generator.canonicalize", canonicalize_answer=lambda v: v.lower())
        _module(ctx, "evals.graders.schema", JudgeDecision=FakeDecision)
        _module(ctx, "evals.graders.judge_prompts", SINGLE_RUBRIC_QA_GROUNDED="test")
        _module(ctx, "evals.graders.runner", GraderResult=object,
                run_config_graders=lambda *a, **kw: [], summarize_graders=lambda *a, **kw: "")
        calls = []

        def answer_query(**kwargs):
            calls.append(kwargs)
            assert "capture_retrieval" not in kwargs
            return "alpha"

        _module(ctx, "retrieval.pipeline", answer_query=answer_query)
        _module(ctx, "stores.base", make_store_from_config=lambda cfg: object())
        judge = _load_source("_test_single_pass_default", "evals/runners/local_judge.py")
        result = judge.run_local_judge(_cfg(), "offline.jsonl")
        assert result["rows"][0]["answer"] == "alpha"
        assert len(calls) == 1


def _load_harness(monkeypatch, rows, judge_runner):
    _setup_common(monkeypatch, rows)
    _module(monkeypatch, "evals.runners.local_judge", run_local_judge=judge_runner)
    _module(monkeypatch, "evals.runners.openai_evals",
            run_openai_evals=lambda *a, **kw: pytest.fail("hosted evals unexpectedly called"))
    return _load_source("_test_single_pass_harness", "evals/harness.py")


def test_harness_uses_judge_answers_once_for_markdown_even_with_duplicate_ids(
    monkeypatch, tmp_path
):
    rows = _rows()
    calls = []
    with monkeypatch.context() as ctx:
        ctx.chdir(tmp_path)

        def judged(cfg, dataset_path, config_path=None):
            calls.append(dataset_path)
            return {
                "rows": [
                    {"question": "slow question", "answer": "alpha", "latency_ms": 90},
                    {"question": "fast question", "answer": "beta", "latency_ms": 15},
                ],
                "pass_rate": 1.0, "total_ms": 120,
            }

        harness = _load_harness(ctx, rows, judged)
        captured = {}
        harness.write_markdown_report = lambda report: captured.setdefault("md", report) and "report.md"
        harness.write_html_report = lambda *a: None
        harness.run_evals(_cfg())

        assert calls == ["offline.jsonl"]
        assert captured["md"]["total_ms"] == 120
        assert [r["answer"] for r in captured["md"]["rows"]] == ["alpha", "beta"]
        assert [r["latency_ms"] for r in captured["md"]["rows"]] == [90, 15]
        assert all(r["em"] == 1.0 and r["f1"] == 1.0 for r in captured["md"]["rows"])


def test_harness_captures_scoring_on_same_pass_and_excludes_raw_text_from_files(
    monkeypatch, tmp_path
):
    rows = _rows()
    calls = []
    sensitive = "do not write this source text to reports"
    with monkeypatch.context() as ctx:
        ctx.chdir(tmp_path)

        def judged(cfg, dataset_path, config_path=None, *, trace_sink):
            calls.append(dataset_path)
            trace_sink(1, {"status": "no_retrieval_call"})
            trace_sink(0, {"status": "captured", "retrieved": [
                {"doc_id": "gold", "text": sensitive},
            ]})
            return {
                "rows": [
                    {"question": "slow question", "answer": "alpha", "latency_ms": 10},
                    {"question": "fast question", "answer": "beta", "latency_ms": 7},
                ],
                "pass_rate": 1.0, "total_ms": 21,
            }

        harness = _load_harness(ctx, rows, judged)
        harness.write_markdown_report = lambda report: "report.md"
        harness.write_html_report = lambda *a: None
        paths = []
        harness.write_retrieval_reports = lambda report, dest: (
            paths.append(report) or ("scores.json", "scores.md")
        )
        harness.run_evals(_cfg(capture=True))
        assert len(calls) == 1
        assert paths[0]["num_examples"] == 1
        assert paths[0]["averages"]["hit@k"] == 1.0
        assert paths[0]["averages"]["precision@k"] == 0.5
        assert sensitive not in json.dumps(paths)
        coverage = (tmp_path / "evals" / "reports" / "retrieval_coverage.json").read_text()
        assert json.loads(coverage)["scored_examples"] == 1
        assert json.loads(coverage)["skipped_reasons"] == {"missing_gold_document_id": 1}
        assert sensitive not in coverage


def test_harness_fails_closed_if_judge_returns_fewer_rows(monkeypatch, tmp_path):
    with monkeypatch.context() as ctx:
        ctx.chdir(tmp_path)
        harness = _load_harness(
            ctx, _rows(), lambda cfg, dataset_path, config_path=None: {
                "rows": [{"question": "slow question", "answer": "alpha", "latency_ms": 5}],
                "pass_rate": 1.0, "total_ms": 5,
            }
        )
        harness.write_markdown_report = lambda report: pytest.fail("incomplete report saved")
        with pytest.raises(ValueError, match="incomplete evaluation batch"):
            harness.run_evals(_cfg())
