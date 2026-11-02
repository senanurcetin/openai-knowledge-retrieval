from __future__ import annotations

import json
import os
from uuid import uuid4
from typing import Any, Dict, List

from rich import print as rprint

from cli.config import RootConfig
from evals.datasets.schema import load_and_validate_jsonl
from evals.metrics.qa import em_f1
from evals.metrics.retrieval_capture import summarize_captured_retrieval
from evals.reporters.retrieval_metrics import (
    write_retrieval_reports,
    write_unavailable_retrieval_reports,
)
from evals.reporters.quality_summary import summarize_qa_quality
from evals.reporters import write_markdown_report
from evals.reporters.html import write_html_report
from evals.runners.local_judge import run_local_judge
from evals.runners.openai_evals import run_openai_evals


def _write_openai_report_link(project: str, report_url: str) -> str:
    out_dir = "evals/reports"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "openai_report_link.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# OpenAI Evals Report\n\nProject: {project}\n\nReport: {report_url}\n"
        )
    return path


def run_evals(cfg: RootConfig, run_ablations: bool = False, config_path: str | None = None) -> None:
    mode = getattr(cfg.evals, "mode", "user")

    if mode == "user":
        if not cfg.evals.dataset_path:
            raise SystemExit("evals.dataset_path not configured for user mode")

        # Correlate every artifact created by this invocation without
        # persisting source content or depending on wall-clock uniqueness.
        run_id = uuid4().hex

        # One answer-generation pass is enough for judging, EM/F1 and reports.
        # Traces are returned only through this ephemeral callback; the judge
        # and HTML payloads never contain raw retrieved source chunks.
        capture_enabled = bool(cfg.evals.retrieval_metrics.capture)
        traces: dict[int, dict[str, Any] | None] = {}

        def record_trace(index: int, trace: dict[str, Any] | None) -> None:
            traces[index] = trace

        if capture_enabled:
            local_summary = run_local_judge(
                cfg, cfg.evals.dataset_path, config_path=config_path,
                trace_sink=record_trace,
            )
        else:
            local_summary = run_local_judge(
                cfg, cfg.evals.dataset_path, config_path=config_path
            )
        rprint(
            f"[green]Local judge pass rate[/green]: {local_summary.get('pass_rate'):.3f}"
        )
        rows = load_and_validate_jsonl(cfg.evals.dataset_path)
        judged = local_summary["rows"]
        if len(judged) != len(rows):
            raise ValueError("Local judge returned an incomplete evaluation batch")
        md_rows: List[Dict[str, Any]] = []
        captured_examples: List[Dict[str, Any]] = []
        for index, (ex, judged_row) in enumerate(zip(rows, judged, strict=True)):
            if judged_row.get("question") != ex.item.question:
                raise ValueError("Local judge result order does not match the dataset")
            ans = judged_row.get("answer") or ""
            em, f1 = em_f1(ans, ex.item.correct_answer or "")
            if capture_enabled:
                captured_examples.append({
                    "id": ex.item.id or "",
                    "source_id": ex.item.source_id,
                    "citation_text": ex.item.citation_text,
                    "retrieval_trace": traces.get(index),
                })
            md_rows.append(
                {
                    "id": ex.item.id or "",
                    "latency_ms": judged_row["latency_ms"],
                    "em": em,
                    "f1": f1,
                    "answer": ans,
                }
            )
        total_ms = local_summary["total_ms"]
        md_report = {
            "project": cfg.project,
            "num_examples": len(rows),
            "total_ms": total_ms,
            "avg_latency_ms": sum(r["latency_ms"] for r in md_rows)
            / max(1, len(md_rows)),
            "avg_em": sum(r["em"] for r in md_rows) / max(1, len(md_rows)),
            "avg_f1": sum(r["f1"] for r in md_rows) / max(1, len(md_rows)),
            "rows": md_rows,
        }
        if config_path:
            md_report["config_path"] = config_path
        quality_summary = summarize_qa_quality(
            judged, md_rows, pass_threshold=cfg.evals.thresholds.pass_rate
        )
        md_report["quality_summary"] = quality_summary
        md_report["run_id"] = run_id
        quality_summary["run_id"] = run_id
        # Shared aggregates in both reports guarantee the same one-pass data
        # underlies Markdown, HTML and machine-readable quality metadata.
        local_summary["quality_summary"] = quality_summary
        local_summary["run_id"] = run_id
        report_dir = os.path.join("evals", "reports")
        os.makedirs(report_dir, exist_ok=True)
        with open(
            os.path.join(report_dir, "quality_summary.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(quality_summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        md_path = write_markdown_report(md_report)
        html_path = os.path.join("evals", "reports", "local_summary.html")
        write_html_report(local_summary, html_path)
        rprint(f"[cyan]Wrote reports[/cyan]: {md_path} and {html_path}")

        if capture_enabled:
            report, coverage = summarize_captured_retrieval(
                captured_examples, top_k=cfg.evals.retrieval_metrics.top_k
            )
            coverage["run_id"] = run_id
            if report is not None:
                report["run_id"] = run_id
            report_dir = os.path.join("evals", "reports")
            os.makedirs(report_dir, exist_ok=True)
            coverage_path = os.path.join(report_dir, "retrieval_coverage.json")
            # Persist only coverage counts and ID/score reports. Never write
            # retrieved source text or raw user queries in these outputs.
            with open(coverage_path, "w", encoding="utf-8") as handle:
                json.dump(coverage, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            if report is not None:
                json_path, markdown_path = write_retrieval_reports(report, report_dir)
                rprint(
                    f"[cyan]Retrieval scores[/cyan]: {json_path}, {markdown_path} "
                    f"({coverage['scored_examples']}/{coverage['total_examples']} scored)"
                )
            else:
                write_unavailable_retrieval_reports(
                    report_dir,
                    top_k=cfg.evals.retrieval_metrics.top_k,
                    reason="no_scored_examples",
                    run_id=run_id,
                )
                rprint(
                    "[yellow]No retrieval scores produced[/yellow]: "
                    "score files now explicitly say 'not_scored'; "
                    "see retrieval_coverage.json for skipped reasons."
                )
        else:
            # Preserve the original default (no extra files) on a fresh run.
            # If this directory contains a prior captured evaluation, replace
            # its metrics/coverage with an explicit disabled status rather
            # than leaving previous scores appearing current.
            report_dir = os.path.join("evals", "reports")
            coverage_path = os.path.join(report_dir, "retrieval_coverage.json")
            if os.path.isfile(coverage_path):
                write_unavailable_retrieval_reports(
                    report_dir,
                    top_k=cfg.evals.retrieval_metrics.top_k,
                    reason="capture_disabled",
                    run_id=run_id,
                )
                coverage = {
                    "run_id": run_id,
                    "status": "capture_disabled",
                    "total_examples": len(rows),
                    "scored_examples": 0,
                    "skipped_examples": len(rows),
                    "skipped_reasons": {"capture_disabled": len(rows)},
                }
                with open(coverage_path, "w", encoding="utf-8") as handle:
                    json.dump(coverage, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")

        #   2) OpenAI Evals (optional)
        if getattr(cfg.evals.openai_evals, "enabled", True):
            # OpenAI Evals
            link = run_openai_evals(cfg, cfg.evals.dataset_path)
            if link.get("skipped"):
                rprint(
                    "[yellow]Skipped OpenAI Evals mirror[/yellow]: "
                    "backend does not expose an OpenAI vector store."
                )
            else:
                link_url = link.get("portal_url") or link.get("report_url", "")
                link_path = _write_openai_report_link(cfg.project, link_url)
                rprint(
                    f"[cyan]OpenAI Evals link[/cyan]: {link_url} (saved at {link_path})"
                )
            return

    elif mode == "auto":
        from evals.generator.auto_pipeline import auto_generate_dataset

        rprint("[cyan]Starting auto-generation pipeline...[/cyan]")
        summary, rows = auto_generate_dataset(cfg)
        # Guard: if no rows were generated, stop with actionable guidance instead of failing later
        if not rows:
            raise SystemExit(
                "Auto-evals produced 0 items. Check that your corpus is ingested and accessible.\n"
                "Tips:\n"
                "- Run 'rag ingest --config <cfg>' first.\n"
                "- If using openai_file_search, set OPENAI_API_KEY and vector_store_id.\n"
                "- For offline/local runs, switch to custom.qdrant backend and ensure Qdrant is up.\n"
                "- Tweak evals.auto parameters (scan_docs, items_target) to broaden sampling."
            )
        rprint(
            f"[green]Auto-generated dataset[/green]: {summary.sampled} items → {summary.dataset_path}"
        )
        # Run local judge on the generated dataset
        local_summary = run_local_judge(cfg, summary.dataset_path, config_path=config_path)
        html_path = os.path.join("evals", "reports", "local_summary.html")
        write_html_report(local_summary, html_path)
        rprint(
            f"[cyan]Local judge completed[/cyan]: pass_rate={local_summary.get('pass_rate'):.3f}"
        )
        # Optionally run OpenAI Evals
        if getattr(cfg.evals.openai_evals, "enabled", True):
            link = run_openai_evals(cfg, summary.dataset_path)
            link_url = link.get("portal_url") or link.get("report_url", "")
            link_path = _write_openai_report_link(cfg.project, link_url)
            rprint(f"[cyan]OpenAI Evals link[/cyan]: {link_url} (saved at {link_path})")
        return
    else:
        raise SystemExit(f"Unknown evals.mode: {mode}")
