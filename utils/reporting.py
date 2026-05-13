from pathlib import Path
from typing import Any

from utils.io import _safe_float, _safe_int, _write_json, _write_jsonl


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


def _write_model_report_artifacts(summary: dict[str, Any], result_file: Path) -> None:
    """Write secondary report artifacts alongside the main result JSON.

    Layout (2026-05 cleanup — markdown renderings removed, traces split):
      <stem>.json                   — main result (details now WITHOUT
                                      interaction_trace; trace_steps count
                                      kept as a compact summary)
      <stem>.summary.json           — main minus details (quick metadata)
      <stem>.details.jsonl          — one detail per line (no full trace)
      <stem>.traces.jsonl           — per-query interaction_trace, one per
                                      line; join key = idx
      <stem>.failures_topk.jsonl    — bottom 30 by judge score
      <stem>.stage_diagnostics.json — execution-stage call rates

    Markdown variants (.summary.md / .failures_topk.md / .stage_diagnostics.md)
    are no longer generated — derived from JSON on demand.
    """
    details = summary.get("details", [])
    if not isinstance(details, list):
        details = []

    stem = result_file.stem
    overview = {key: value for key, value in summary.items() if key != "details"}

    summary_json_file = result_file.with_name(f"{stem}.summary.json")
    details_jsonl_file = result_file.with_name(f"{stem}.details.jsonl")
    traces_jsonl_file = result_file.with_name(f"{stem}.traces.jsonl")
    failures_jsonl_file = result_file.with_name(f"{stem}.failures_topk.jsonl")
    stage_diag_json_file = result_file.with_name(f"{stem}.stage_diagnostics.json")

    _write_json(summary_json_file, overview)

    detail_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(details, start=1):
        trace = item.get("interaction_trace", [])
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
            "trace_steps": _collect_trace_steps(trace),
        })
        trace_rows.append({
            "idx": idx,
            "query": item.get("query", ""),
            "interaction_trace": trace,
        })
    _write_jsonl(details_jsonl_file, detail_rows)
    _write_jsonl(traces_jsonl_file, trace_rows)

    failures = _build_failure_records(details, top_k=30)
    _write_jsonl(failures_jsonl_file, failures)

    diagnostics = _compute_stage_diagnostics(details)
    _write_json(stage_diag_json_file, diagnostics)
