from __future__ import annotations

import os
from datetime import datetime, timezone
from html import escape
from typing import Any, Dict


def _safe_markdown(value: object) -> str:
    """Keep untrusted content inside one Markdown cell or heading line."""
    raw = "" if value is None else str(value)
    safe = escape(raw, quote=True)
    # Entities avoid table delimiters, code/link Markdown and HTML injection.
    for char, entity in (("|", "&#124;"), ("`", "&#96;"), ("[", "&#91;"), ("]", "&#93;")):
        safe = safe.replace(char, entity)
    return safe.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def write_markdown_report(
    report: Dict[str, Any], out_dir: str = "evals/reports"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"report_{ts}.md")
    lines = []
    lines.append(f"# RAG Eval Report - {_safe_markdown(report.get('project'))}")
    lines.append("")
    if report.get("config_path"):
        lines.append(f"Config: `{_safe_markdown(report['config_path'])}`  ")
    lines.append(f"Examples: {report.get('num_examples')}  ")
    lines.append(f"Avg latency: {report.get('avg_latency_ms'):.1f} ms  ")
    lines.append(f"EM: {report.get('avg_em'):.3f}  F1: {report.get('avg_f1'):.3f}")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("| id | latency_ms | em | f1 | answer |")
    lines.append("|----|------------:|---:|---:|--------|")
    for r in report.get("rows", []):
        lines.append(
            f"| {_safe_markdown(r['id'])} | {r['latency_ms']} | {r['em']:.1f} "
            f"| {r['f1']:.3f} | {_safe_markdown(r['answer'])} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
