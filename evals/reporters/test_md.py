"""Offline tests for safely rendering untrusted local evaluation answers."""
from pathlib import Path
from evals.reporters.md import write_markdown_report


def test_markdown_report_escapes_untrusted_answers_and_keeps_one_row(tmp_path):
    report = {
        "project": "Example\n# Forged heading",
        "num_examples": 1,
        "avg_latency_ms": 4.0,
        "avg_em": 0.0,
        "avg_f1": 0.4,
        "rows": [{
            "id": "row|1",
            "latency_ms": 4,
            "em": 0.0,
            "f1": 0.4,
            "answer": "<script>alert(1)</script>\n| fake | row |",
        }],
    }
    output = Path(write_markdown_report(report, out_dir=str(tmp_path))).read_text(encoding="utf-8")
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert "row&#124;1" in output
    assert "&#124; fake &#124; row &#124;" in output
    assert "\n| fake | row |" not in output
    assert "\n# Forged heading" not in output
    assert len([line for line in output.splitlines() if "row&#124;1" in line]) == 1


def test_markdown_report_accepts_null_answer_without_crashing(tmp_path):
    report = {
        "project": "Demo", "num_examples": 1, "avg_latency_ms": 0.0,
        "avg_em": 0.0, "avg_f1": 0.0,
        "rows": [{"id": None, "latency_ms": 0, "em": 0, "f1": 0, "answer": None}],
    }
    path = Path(write_markdown_report(report, out_dir=str(tmp_path)))
    assert path.exists()
    assert "| 0 |" in path.read_text(encoding="utf-8")
