from pathlib import Path
from typing import Any

from utils.io import _safe_float, _safe_int, _to_markdown_table, _write_json, _write_jsonl


def _collect_trace_steps(trace: Any) -> list[str]:
    if not isinstance(trace, list):
        return []
    steps = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "") or "").strip()
        if step:
            steps.append(step)
    return steps


def _is_insufficient_answer_text(answer: Any) -> bool:
    return "insufficient evidence" in str(answer or "").lower()


def _is_runtime_error_row(item: dict[str, Any]) -> bool:
    if bool(item.get("error")):
        return True
    answer_text = str(item.get("answer", "") or "").lower()
    return answer_text.startswith("@@answer: error")


def _compute_stage_diagnostics(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    if total == 0:
        return {
            "queries": 0,
            "answer_attempt_count": 0,
            "answer_attempt_rate": 0.0,
            "hallucination_count": 0,
            "hallucination_rate": 0.0,
            "hallucination_eligible_count": 0,
            "hallucination_rate_answered": 0.0,
            "insufficient_count": 0,
            "forced_synthesis_count": 0,
            "compute_missing_guard_count": 0,
            "reflection_count": 0,
            "refinement_count": 0,
            "avg_reflection_attempts": 0.0,
            "avg_refinement_attempts": 0.0,
            "avg_synthesis_attempts": 0.0,
        }

    answer_attempt_count = 0
    hallucination_count = 0
    hallucination_eligible_count = 0
    insufficient_count = 0
    forced_synthesis_count = 0
    compute_missing_guard_count = 0
    reflection_count = 0
    refinement_count = 0
    reflection_attempts_sum = 0
    refinement_attempts_sum = 0
    synthesis_attempts_sum = 0

    for item in details:
        is_insufficient = _is_insufficient_answer_text(item.get("answer", ""))
        if is_insufficient:
            insufficient_count += 1
        else:
            is_error = _is_runtime_error_row(item)
            if not is_error:
                answer_attempt_count += 1
                hallucination_value = item.get("hallucination", None)
                if isinstance(hallucination_value, (int, float)):
                    hallucination_eligible_count += 1
                    if _safe_float(hallucination_value, 0.0) >= 0.5:
                        hallucination_count += 1
                else:
                    score_value = item.get("llm_judge_score", None)
                    if isinstance(score_value, (int, float)):
                        hallucination_eligible_count += 1
                        if _safe_float(score_value, 0.0) < 1.0:
                            hallucination_count += 1

        trace = item.get("interaction_trace", [])
        if not isinstance(trace, list):
            trace = []

        per_query_reflections = 0
        for event in trace:
            if not isinstance(event, dict):
                continue
            step = str(event.get("step", "") or "")
            output = event.get("output", {})
            if step == "execution_forced_synthesis":
                forced_synthesis_count += 1
                if isinstance(output, dict) and isinstance(output.get("attempts"), list):
                    synthesis_attempts_sum += len(output.get("attempts", []))
                else:
                    synthesis_attempts_sum += 1
            elif step == "execution_compute_missing_guard":
                compute_missing_guard_count += 1
            elif step == "reflection":
                reflection_count += 1
                per_query_reflections += 1
            elif step == "refinement":
                refinement_count += 1
                if isinstance(output, dict) and isinstance(output.get("attempts"), list):
                    refinement_attempts_sum += len(output.get("attempts", []))
                else:
                    refinement_attempts_sum += 1
        reflection_attempts_sum += per_query_reflections

    return {
        "queries": total,
        "answer_attempt_count": answer_attempt_count,
        "answer_attempt_rate": answer_attempt_count / total,
        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_count / total,
        "hallucination_eligible_count": hallucination_eligible_count,
        "hallucination_rate_answered": hallucination_count / max(1, hallucination_eligible_count),
        "insufficient_count": insufficient_count,
        "insufficient_rate": insufficient_count / total,
        "forced_synthesis_count": forced_synthesis_count,
        "forced_synthesis_rate": forced_synthesis_count / total,
        "compute_missing_guard_count": compute_missing_guard_count,
        "compute_missing_guard_rate": compute_missing_guard_count / total,
        "reflection_count": reflection_count,
        "reflection_rate": reflection_count / total,
        "refinement_count": refinement_count,
        "refinement_rate": refinement_count / total,
        "avg_reflection_attempts": reflection_attempts_sum / total,
        "avg_refinement_attempts": refinement_attempts_sum / total,
        "avg_synthesis_attempts": synthesis_attempts_sum / max(1, forced_synthesis_count),
    }


def _build_model_summary_markdown(summary: dict[str, Any], result_filename: str) -> str:
    avg_fields = [key for key in sorted(summary.keys()) if key.startswith("avg_")]
    metric_rows: list[list[Any]] = []
    for key in avg_fields:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            metric_rows.append([key, f"{float(value):.4f}"])
    metric_table = _to_markdown_table(["metric", "value"], metric_rows) if metric_rows else "_No numeric metrics_"

    cat_rows: list[list[Any]] = []
    cat_summaries = summary.get("category_summaries", {})
    if isinstance(cat_summaries, dict):
        for cat, cat_info in sorted(cat_summaries.items()):
            if not isinstance(cat_info, dict):
                continue
            cat_rows.append([
                cat,
                cat_info.get("count", 0),
                f"{_safe_float(cat_info.get('avg_llm_judge_score', 0.0)):.4f}",
                f"{_safe_float(cat_info.get('avg_hallucination', 0.0)):.4f}",
                f"{_safe_float(cat_info.get('avg_answer_attempted', 0.0)):.4f}",
                f"{_safe_float(cat_info.get('avg_doc_match', 0.0)):.4f}",
                f"{_safe_float(cat_info.get('avg_page_match', 0.0)):.4f}",
                f"{_safe_float(cat_info.get('avg_latency', 0.0)):.4f}",
            ])
    cat_table = _to_markdown_table(
        [
            "category",
            "count",
            "avg_llm_judge_score",
            "avg_hallucination",
            "avg_answer_attempted",
            "avg_doc_match",
            "avg_page_match",
            "avg_latency",
        ],
        cat_rows,
    ) if cat_rows else "_No category rows_"

    lines = [
        f"# Benchmark Summary: {result_filename}",
        "",
        f"- strategy: `{summary.get('strategy', '')}`",
        f"- corpus_tag: `{summary.get('corpus_tag', '')}`",
        f"- dataset: `{summary.get('dataset', '')}`",
        f"- status: `{summary.get('status', '')}`",
        f"- queries_count: `{summary.get('queries_count', 0)}` / `{summary.get('total_queries', 0)}`",
        "",
        "## Overall Metrics",
        "",
        metric_table,
        "",
        "## Category Breakdown",
        "",
        cat_table,
    ]
    return "\n".join(lines) + "\n"


def _build_stage_diagnostics_markdown(diag: dict[str, Any], result_filename: str) -> str:
    rows = [
        ["queries", _safe_int(diag.get("queries", 0))],
        ["answer_attempt_count", _safe_int(diag.get("answer_attempt_count", 0))],
        ["answer_attempt_rate", f"{_safe_float(diag.get('answer_attempt_rate', 0.0)):.4f}"],
        ["hallucination_count", _safe_int(diag.get("hallucination_count", 0))],
        ["hallucination_rate", f"{_safe_float(diag.get('hallucination_rate', 0.0)):.4f}"],
        ["hallucination_eligible_count", _safe_int(diag.get("hallucination_eligible_count", 0))],
        ["hallucination_rate_answered", f"{_safe_float(diag.get('hallucination_rate_answered', 0.0)):.4f}"],
        ["insufficient_count", _safe_int(diag.get("insufficient_count", 0))],
        ["insufficient_rate", f"{_safe_float(diag.get('insufficient_rate', 0.0)):.4f}"],
        ["forced_synthesis_count", _safe_int(diag.get("forced_synthesis_count", 0))],
        ["forced_synthesis_rate", f"{_safe_float(diag.get('forced_synthesis_rate', 0.0)):.4f}"],
        ["compute_missing_guard_count", _safe_int(diag.get("compute_missing_guard_count", 0))],
        ["compute_missing_guard_rate", f"{_safe_float(diag.get('compute_missing_guard_rate', 0.0)):.4f}"],
        ["reflection_count", _safe_int(diag.get("reflection_count", 0))],
        ["reflection_rate", f"{_safe_float(diag.get('reflection_rate', 0.0)):.4f}"],
        ["refinement_count", _safe_int(diag.get("refinement_count", 0))],
        ["refinement_rate", f"{_safe_float(diag.get('refinement_rate', 0.0)):.4f}"],
        ["avg_reflection_attempts", f"{_safe_float(diag.get('avg_reflection_attempts', 0.0)):.4f}"],
        ["avg_refinement_attempts", f"{_safe_float(diag.get('avg_refinement_attempts', 0.0)):.4f}"],
        ["avg_synthesis_attempts", f"{_safe_float(diag.get('avg_synthesis_attempts', 0.0)):.4f}"],
    ]
    return "\n".join([
        f"# Stage Diagnostics: {result_filename}",
        "",
        _to_markdown_table(["metric", "value"], rows),
        "",
    ])


def _build_failure_records(details: list[dict[str, Any]], top_k: int = 30) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for idx, item in enumerate(details, start=1):
        score = item.get("llm_judge_score", None)
        has_error = bool(item.get("error"))
        is_failure = has_error
        if score is not None:
            is_failure = is_failure or (_safe_float(score, 0.0) < 1.0)
        if not is_failure:
            continue
        failures.append({
            "rank_hint": idx,
            "query": item.get("query", ""),
            "category": item.get("category", ""),
            "llm_judge_score": _safe_float(score, 0.0),
            "hallucination": _safe_float(item.get("hallucination", 0.0)),
            "hallucination_reason": item.get("hallucination_reason", ""),
            "hallucination_model": item.get("hallucination_model", ""),
            "llm_judge_reason": item.get("llm_judge_reason", ""),
            "doc_match": _safe_float(item.get("doc_match", 0.0)),
            "page_match": _safe_float(item.get("page_match", 0.0)),
            "latency": _safe_float(item.get("latency", 0.0)),
            "answer": item.get("answer", ""),
            "ground_truth": item.get("ground_truth", ""),
            "error": item.get("error", ""),
            "trace_steps": _collect_trace_steps(item.get("interaction_trace", [])),
        })
    failures.sort(key=lambda item: (item.get("llm_judge_score", 0.0), -item.get("doc_match", 0.0), -item.get("latency", 0.0)))
    return failures[: max(1, top_k)]


def _build_failures_markdown(records: list[dict[str, Any]], result_filename: str) -> str:
    rows = []
    for idx, record in enumerate(records, start=1):
        rows.append([
            idx,
            f"{_safe_float(record.get('llm_judge_score', 0.0)):.1f}",
            f"{_safe_float(record.get('hallucination', 0.0)):.1f}",
            f"{_safe_float(record.get('doc_match', 0.0)):.1f}",
            f"{_safe_float(record.get('page_match', 0.0)):.1f}",
            record.get("category", ""),
            record.get("query", ""),
            record.get("llm_judge_reason", ""),
            record.get("hallucination_reason", ""),
        ])
    table = _to_markdown_table(
        ["rank", "judge", "hallu", "doc", "page", "category", "query", "judge_reason", "hallu_reason"],
        rows,
    ) if rows else "_No failures_"
    return "\n".join([
        f"# Failures Top-K: {result_filename}",
        "",
        table,
        "",
    ])


def _write_model_report_artifacts(summary: dict[str, Any], result_file: Path) -> None:
    details = summary.get("details", [])
    if not isinstance(details, list):
        details = []

    stem = result_file.stem
    overview = {key: value for key, value in summary.items() if key != "details"}

    summary_json_file = result_file.with_name(f"{stem}.summary.json")
    summary_md_file = result_file.with_name(f"{stem}.summary.md")
    details_jsonl_file = result_file.with_name(f"{stem}.details.jsonl")
    failures_jsonl_file = result_file.with_name(f"{stem}.failures_topk.jsonl")
    failures_md_file = result_file.with_name(f"{stem}.failures_topk.md")
    stage_diag_json_file = result_file.with_name(f"{stem}.stage_diagnostics.json")
    stage_diag_md_file = result_file.with_name(f"{stem}.stage_diagnostics.md")

    _write_json(summary_json_file, overview)
    summary_md_file.write_text(_build_model_summary_markdown(summary, result_file.name), encoding="utf-8")

    detail_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(details, start=1):
        detail_rows.append({
            "idx": idx,
            "query": item.get("query", ""),
            "category": item.get("category", ""),
            "answer": item.get("answer", ""),
            "ground_truth": item.get("ground_truth", ""),
            "llm_judge_score": _safe_float(item.get("llm_judge_score", 0.0)),
            "answer_attempted": _safe_float(item.get("answer_attempted", 0.0)),
            "hallucination": _safe_float(item.get("hallucination", 0.0)),
            "hallucination_reason": item.get("hallucination_reason", ""),
            "hallucination_source": item.get("hallucination_source", ""),
            "hallucination_model": item.get("hallucination_model", ""),
            "llm_judge_reason": item.get("llm_judge_reason", ""),
            "doc_match": _safe_float(item.get("doc_match", 0.0)),
            "page_match": _safe_float(item.get("page_match", 0.0)),
            "latency": _safe_float(item.get("latency", 0.0)),
            "error": item.get("error", ""),
            "trace_steps": _collect_trace_steps(item.get("interaction_trace", [])),
        })
    _write_jsonl(details_jsonl_file, detail_rows)

    failures = _build_failure_records(details, top_k=30)
    _write_jsonl(failures_jsonl_file, failures)
    failures_md_file.write_text(_build_failures_markdown(failures, result_file.name), encoding="utf-8")

    diagnostics = _compute_stage_diagnostics(details)
    _write_json(stage_diag_json_file, diagnostics)
    stage_diag_md_file.write_text(
        _build_stage_diagnostics_markdown(diagnostics, result_file.name),
        encoding="utf-8",
    )
