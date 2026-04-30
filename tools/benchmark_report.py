#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple


RESULTS_ROOT = Path("data/results")
INDEX_PATH = RESULTS_ROOT / "index.jsonl"
PARALLEL_ALL_EXPECTED_VARIANTS = [
    "naive::naive::refl_on::agentic_off",
    "hoprag::hoprag_full::refl_on::agentic_off",
    "ms_graphrag::ms_graphrag_full::refl_on::agentic_off",
    "hyporeflect::full::refl_on::agentic_off",
    "naive::naive_no_table::refl_on::agentic_off",
    "naive::naive_no_chunk::refl_on::agentic_off",
    "naive::naive_no_summary::refl_on::agentic_off",
    "hyporeflect::no_table::refl_on::agentic_off",
    "hyporeflect::no_chunk::refl_on::agentic_off",
    "hyporeflect::no_summary::refl_on::agentic_off",
    "hoprag::hoprag_no_table::refl_on::agentic_off",
    "hoprag::hoprag_no_chunk::refl_on::agentic_off",
    "hoprag::hoprag_no_summary::refl_on::agentic_off",
    "ms_graphrag::ms_graphrag_no_table::refl_on::agentic_off",
    "ms_graphrag::ms_graphrag_no_chunk::refl_on::agentic_off",
    "ms_graphrag::ms_graphrag_no_summary::refl_on::agentic_off",
]


def _parallel_all_agentic_expected_variants() -> List[str]:
    rows: List[str] = []
    for mode in ("agentic_off", "agentic_on"):
        rows.extend(
            [
                f"naive::naive::refl_on::{mode}",
                f"naive::naive_no_table::refl_on::{mode}",
                f"naive::naive_no_chunk::refl_on::{mode}",
                f"naive::naive_no_summary::refl_on::{mode}",
                f"hoprag::hoprag_full::refl_on::{mode}",
                f"hoprag::hoprag_no_table::refl_on::{mode}",
                f"hoprag::hoprag_no_chunk::refl_on::{mode}",
                f"hoprag::hoprag_no_summary::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_full::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_no_table::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_no_chunk::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_no_summary::refl_on::{mode}",
                f"hyporeflect::full::refl_on::{mode}",
                f"hyporeflect::no_table::refl_on::{mode}",
                f"hyporeflect::no_chunk::refl_on::{mode}",
                f"hyporeflect::no_summary::refl_on::{mode}",
            ]
        )
    return rows


def _agentic_matrix_expected_variants() -> List[str]:
    rows: List[str] = []
    for mode in ("agentic_off", "agentic_on"):
        rows.extend(
            [
                f"naive::naive::refl_on::{mode}",
                f"naive::naive_no_table::refl_on::{mode}",
                f"naive::naive_no_chunk::refl_on::{mode}",
                f"naive::naive_no_summary::refl_on::{mode}",
                f"hoprag::hoprag_full::refl_on::{mode}",
                f"hoprag::hoprag_no_table::refl_on::{mode}",
                f"hoprag::hoprag_no_chunk::refl_on::{mode}",
                f"hoprag::hoprag_no_summary::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_full::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_no_table::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_no_chunk::refl_on::{mode}",
                f"ms_graphrag::ms_graphrag_no_summary::refl_on::{mode}",
                f"hyporeflect::full::refl_on::{mode}",
                f"hyporeflect::no_table::refl_on::{mode}",
                f"hyporeflect::no_chunk::refl_on::{mode}",
                f"hyporeflect::no_summary::refl_on::{mode}",
            ]
        )
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_insufficient_answer_text(answer: Any) -> bool:
    return "insufficient evidence" in str(answer or "").lower()


def _is_runtime_error_row(item: Dict[str, Any]) -> bool:
    if bool(item.get("error")):
        return True
    answer_text = str(item.get("answer", "") or "").lower()
    return answer_text.startswith("@@answer: error")


def _compute_hallucination_stats(details: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(details)
    if total == 0:
        return {
            "answer_attempt_count": 0.0,
            "answer_attempt_rate": 0.0,
            "hallucination_count": 0.0,
            "hallucination_rate": 0.0,
            "hallucination_eligible_count": 0.0,
            "hallucination_rate_answered": 0.0,
        }

    answer_attempt_count = 0
    hallucination_count = 0
    hallucination_eligible_count = 0
    for item in details:
        if _is_insufficient_answer_text(item.get("answer", "")):
            continue
        if _is_runtime_error_row(item):
            continue
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

    return {
        "answer_attempt_count": float(answer_attempt_count),
        "answer_attempt_rate": answer_attempt_count / total,
        "hallucination_count": float(hallucination_count),
        "hallucination_rate": hallucination_count / total,
        "hallucination_eligible_count": float(hallucination_eligible_count),
        "hallucination_rate_answered": hallucination_count / max(1, hallucination_eligible_count),
    }


def _escape_md(text: Any) -> str:
    return str(text or "").replace("\n", " ").replace("|", "\\|").strip()


def _to_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_escape_md(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _enable_reflection(payload: Dict[str, Any]) -> bool:
    ablation = payload.get("ablation")
    if isinstance(ablation, dict) and "enable_reflection" in ablation:
        return _as_bool(ablation.get("enable_reflection"), default=True)
    return True


def _variant_reflection_token(payload: Dict[str, Any]) -> str:
    return "refl_on" if _enable_reflection(payload) else "refl_off"


def _variant_agentic_token(payload: Dict[str, Any]) -> str:
    mode = str(payload.get("agentic_mode", "") or "").strip().lower()
    if mode in {"on", "off"}:
        return f"agentic_{mode}"
    return ""


def _variant_id(strategy: str, corpus_tag: str, reflection_token: str, agentic_token: str = "") -> str:
    base = f"{strategy}::{corpus_tag}::{reflection_token}"
    if agentic_token:
        return f"{base}::{agentic_token}"
    return base


def _split_variant_id(key: str) -> Tuple[str, str, str, str]:
    parts = str(key or "").split("::")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], "::".join(parts[3:])
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], ""
    if len(parts) == 2:
        return parts[0], parts[1], "", ""
    if len(parts) == 1:
        return parts[0], "", "", ""
    return "", "", "", ""


def _is_primary_result_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if not payload.get("strategy"):
        return False
    if not isinstance(payload.get("details"), list):
        return False
    if not payload.get("dataset"):
        return False
    return True


def _iter_primary_results(run_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    rows: List[Tuple[Path, Dict[str, Any]]] = []
    for fp in sorted(run_dir.rglob("*.json")):
        name = fp.name
        if name.endswith(".summary.json"):
            continue
        if "stage_diagnostics" in name:
            continue
        if name.startswith("run_"):
            continue
        payload = _load_json(fp)
        if _is_primary_result_payload(payload):
            rows.append((fp, payload))
    return rows


def _collect_trace_steps(trace: Any) -> List[str]:
    if not isinstance(trace, list):
        return []
    steps: List[str] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "") or "").strip()
        if step:
            steps.append(step)
    return steps


def _compute_stage_diagnostics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _result_row(run_id: str, fp: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    gate = payload.get("benchmark_gate", {})
    gate_passed = None
    if isinstance(gate, dict):
        gate_passed = gate.get("passed")

    strategy = str(payload.get("strategy", "") or "")
    corpus_tag = str(payload.get("corpus_tag", "") or "")
    reflection_token = _variant_reflection_token(payload)
    agentic_token = _variant_agentic_token(payload)
    agentic_mode = str(payload.get("agentic_mode", "") or "")
    details = payload.get("details", [])
    if not isinstance(details, list):
        details = []
    halluc_stats = _compute_hallucination_stats(details)
    avg_hallucination = payload.get("avg_hallucination", None)
    if avg_hallucination is None and details:
        avg_hallucination = halluc_stats.get("hallucination_rate", 0.0)
    avg_answer_attempted = payload.get("avg_answer_attempted", None)
    if avg_answer_attempted is None and details:
        avg_answer_attempted = halluc_stats.get("answer_attempt_rate", 0.0)
    return {
        "run_id": run_id,
        "run_dir": str(fp.parent),
        "result_file": str(fp),
        "result_name": fp.name,
        "strategy": strategy,
        "corpus_tag": corpus_tag,
        "enable_reflection": _enable_reflection(payload),
        "reflection_token": reflection_token,
        "agentic_mode": agentic_mode,
        "agentic_token": agentic_token,
        "variant_id": _variant_id(strategy, corpus_tag, reflection_token, agentic_token),
        "dataset": str(payload.get("dataset", "") or ""),
        "status": str(payload.get("status", "") or ""),
        "queries_count": _safe_int(payload.get("queries_count", 0)),
        "total_queries": _safe_int(payload.get("total_queries", 0)),
        "avg_latency": _safe_float(payload.get("avg_latency", 0.0)),
        "avg_llm_judge_score": _safe_float(payload.get("avg_llm_judge_score", 0.0)),
        "avg_hallucination": _safe_float(avg_hallucination, 0.0),
        "avg_answer_attempted": _safe_float(avg_answer_attempted, 0.0),
        "avg_doc_match": _safe_float(payload.get("avg_doc_match", 0.0)),
        "avg_page_match": _safe_float(payload.get("avg_page_match", 0.0)),
        "timestamp": str(payload.get("timestamp", "") or ""),
        "gate_passed": gate_passed,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception:
                continue
    return rows


def _collect_failures_for_run(run_items: List[Tuple[Path, Dict[str, Any]]], top_k: int = 50) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for fp, payload in run_items:
        model_key = _variant_id(
            str(payload.get("strategy", "") or ""),
            str(payload.get("corpus_tag", "") or ""),
            _variant_reflection_token(payload),
            _variant_agentic_token(payload),
        )
        details = payload.get("details", [])
        if not isinstance(details, list):
            continue
        for idx, item in enumerate(details, start=1):
            has_error = bool(item.get("error"))
            score = item.get("llm_judge_score", None)
            answer_text = str(item.get("answer", "") or "")
            is_insufficient = _is_insufficient_answer_text(answer_text)
            hallucination_value = item.get("hallucination", None)
            if not isinstance(hallucination_value, (int, float)):
                hallucination_value = 1.0 if (not is_insufficient and not has_error and _safe_float(score, 0.0) < 1.0) else 0.0
            is_failure = has_error
            if score is not None:
                is_failure = is_failure or (_safe_float(score, 0.0) < 1.0)
            if not is_failure:
                continue
            failures.append({
                "model": model_key,
                "result_name": fp.name,
                "idx": idx,
                "query": item.get("query", ""),
                "category": item.get("category", ""),
                "llm_judge_score": _safe_float(score, 0.0),
                "hallucination": _safe_float(hallucination_value, 0.0),
                "hallucination_reason": item.get("hallucination_reason", ""),
                "hallucination_model": item.get("hallucination_model", ""),
                "llm_judge_reason": item.get("llm_judge_reason", ""),
                "doc_match": _safe_float(item.get("doc_match", 0.0)),
                "page_match": _safe_float(item.get("page_match", 0.0)),
                "latency": _safe_float(item.get("latency", 0.0)),
                "error": item.get("error", ""),
                "trace_steps": _collect_trace_steps(item.get("interaction_trace", [])),
            })
    failures.sort(key=lambda x: (x.get("llm_judge_score", 0.0), -x.get("doc_match", 0.0), -x.get("latency", 0.0)))
    return failures[: max(1, top_k)]


def _build_global_index(results_root: Path) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    if not results_root.exists():
        return all_rows
    for run_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        run_id = run_dir.name
        for fp, payload in _iter_primary_results(run_dir):
            all_rows.append(_result_row(run_id, fp, payload))
    all_rows.sort(key=lambda x: (x.get("run_id", ""), x.get("strategy", ""), x.get("corpus_tag", "")), reverse=True)
    _write_jsonl(INDEX_PATH, all_rows)
    return all_rows


def _resolve_expected_variants(profile: str) -> List[str]:
    token = str(profile or "").strip().lower()
    if token in {"parallel_all", "all"}:
        return list(PARALLEL_ALL_EXPECTED_VARIANTS)
    if token in {"parallel_all_agentic", "all_agentic"}:
        return _parallel_all_agentic_expected_variants()
    if token in {"agentic_matrix", "agentic"}:
        return _agentic_matrix_expected_variants()
    return []


def _coverage_summary(rows: List[Dict[str, Any]], expected_variants: List[str]) -> Dict[str, Any]:
    present_ids = [str(r.get("variant_id", "") or "") for r in rows if r.get("variant_id")]
    present_set = set(present_ids)
    expected_set = set(expected_variants)
    missing = sorted(expected_set - present_set)
    unexpected = sorted(present_set - expected_set) if expected_set else sorted(present_set)
    duplicate_counter = Counter(present_ids)
    duplicates = sorted([key for key, count in duplicate_counter.items() if count > 1])
    return {
        "expected_count": len(expected_variants),
        "present_count": len(present_ids),
        "present_unique_count": len(present_set),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
    }


def _write_coverage_artifacts(run_dir: Path, run_id: str, rows: List[Dict[str, Any]], profile: str) -> Dict[str, Any]:
    expected_variants = _resolve_expected_variants(profile)
    coverage = _coverage_summary(rows, expected_variants)
    payload = {
        "run_id": run_id,
        "profile": profile,
        "expected_variants": expected_variants,
        **coverage,
    }
    _write_json(run_dir / "run_coverage.json", payload)
    md_lines = [
        f"# Run Coverage: {run_id}",
        "",
        f"- profile: `{profile}`",
        f"- expected_count: `{coverage['expected_count']}`",
        f"- present_count: `{coverage['present_count']}`",
        f"- present_unique_count: `{coverage['present_unique_count']}`",
        f"- missing_count: `{len(coverage['missing'])}`",
        f"- unexpected_count: `{len(coverage['unexpected'])}`",
        f"- duplicate_variant_count: `{len(coverage['duplicates'])}`",
        "",
    ]

    if coverage["missing"]:
        md_lines.extend(["## Missing", ""])
        md_lines.extend([f"- `{m}`" for m in coverage["missing"]])
        md_lines.append("")

    if coverage["duplicates"]:
        md_lines.extend(["## Duplicates", ""])
        md_lines.extend([f"- `{d}`" for d in coverage["duplicates"]])
        md_lines.append("")

    if coverage["unexpected"]:
        md_lines.extend(["## Unexpected", ""])
        md_lines.extend([f"- `{u}`" for u in coverage["unexpected"]])
        md_lines.append("")

    if not coverage["missing"] and not coverage["duplicates"] and not coverage["unexpected"]:
        md_lines.append("_Coverage is clean._")
        md_lines.append("")

    (run_dir / "run_coverage.md").write_text("\n".join(md_lines), encoding="utf-8")
    return payload


def cmd_generate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"run_dir not found: {run_dir}")
        return 1

    run_items = _iter_primary_results(run_dir)
    if not run_items:
        print(f"No benchmark result json found in {run_dir}")
        return 1

    rows = [_result_row(run_dir.name, fp, payload) for fp, payload in run_items]
    rows.sort(key=lambda x: (x.get("avg_llm_judge_score", 0.0), x.get("avg_doc_match", 0.0), -x.get("avg_latency", 0.0)), reverse=True)

    leaderboard_json = run_dir / "run_leaderboard.json"
    leaderboard_md = run_dir / "run_leaderboard.md"
    _write_json(leaderboard_json, {"run_id": run_dir.name, "models": rows})

    board_rows = []
    for r in rows:
        board_rows.append([
            r.get("strategy", ""),
            r.get("corpus_tag", ""),
            r.get("reflection_token", ""),
            r.get("agentic_token", ""),
            r.get("status", ""),
            f"{_safe_float(r.get('avg_llm_judge_score', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_hallucination', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_answer_attempted', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_doc_match', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_page_match', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_latency', 0.0)):.4f}",
            f"{_safe_int(r.get('queries_count', 0))}/{_safe_int(r.get('total_queries', 0))}",
            r.get("gate_passed", ""),
            r.get("result_name", ""),
        ])
    leaderboard_md.write_text(
        "\n".join([
            f"# Run Leaderboard: {run_dir.name}",
            "",
            _to_markdown_table(
                [
                    "strategy",
                    "corpus_tag",
                    "reflection",
                    "agentic",
                    "status",
                    "avg_llm_judge",
                    "avg_hallucination",
                    "avg_answer_attempted",
                    "avg_doc",
                    "avg_page",
                    "avg_latency",
                    "progress",
                    "gate",
                    "file",
                ],
                board_rows,
            ),
            "",
        ]),
        encoding="utf-8",
    )

    failures = _collect_failures_for_run(run_items, top_k=args.top_k)
    failures_jsonl = run_dir / "run_failures_topk.jsonl"
    failures_md = run_dir / "run_failures_topk.md"
    _write_jsonl(failures_jsonl, failures)
    failure_rows = []
    for i, f in enumerate(failures, start=1):
        failure_rows.append([
            i,
            f.get("model", ""),
            f"{_safe_float(f.get('llm_judge_score', 0.0)):.1f}",
            f"{_safe_float(f.get('hallucination', 0.0)):.1f}",
            f"{_safe_float(f.get('doc_match', 0.0)):.1f}",
            f"{_safe_float(f.get('page_match', 0.0)):.1f}",
            f.get("query", ""),
            f.get("llm_judge_reason", ""),
            f.get("hallucination_reason", ""),
        ])
    failures_md.write_text(
        "\n".join([
            f"# Run Failures Top-{args.top_k}: {run_dir.name}",
            "",
            _to_markdown_table(["rank", "model", "judge", "hallu", "doc", "page", "query", "judge_reason", "hallu_reason"], failure_rows) if failure_rows else "_No failures_",
            "",
        ]),
        encoding="utf-8",
    )

    diag_rows = []
    for fp, payload in run_items:
        details = payload.get("details", [])
        if not isinstance(details, list):
            details = []
        diag = _compute_stage_diagnostics(details)
        diag_rows.append({
            "strategy": payload.get("strategy", ""),
            "corpus_tag": payload.get("corpus_tag", ""),
            "reflection_token": _variant_reflection_token(payload),
            "agentic_token": _variant_agentic_token(payload),
            **diag,
            "result_name": fp.name,
        })
    _write_json(run_dir / "run_stage_diagnostics.json", {"run_id": run_dir.name, "models": diag_rows})
    stage_rows = []
    for d in diag_rows:
        stage_rows.append([
            d.get("strategy", ""),
            d.get("corpus_tag", ""),
            d.get("reflection_token", ""),
            d.get("agentic_token", ""),
            _safe_int(d.get("queries", 0)),
            f"{_safe_float(d.get('answer_attempt_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('hallucination_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('hallucination_rate_answered', 0.0)):.4f}",
            f"{_safe_float(d.get('insufficient_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('forced_synthesis_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('compute_missing_guard_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('reflection_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('refinement_rate', 0.0)):.4f}",
            f"{_safe_float(d.get('avg_refinement_attempts', 0.0)):.4f}",
        ])
    (run_dir / "run_stage_diagnostics.md").write_text(
        "\n".join([
            f"# Run Stage Diagnostics: {run_dir.name}",
            "",
            _to_markdown_table(
                [
                    "strategy",
                    "corpus_tag",
                    "reflection",
                    "agentic",
                    "queries",
                    "answer_attempt_rate",
                    "hallucination_rate",
                    "hallucination_rate_answered",
                    "insufficient_rate",
                    "forced_synthesis_rate",
                    "compute_missing_guard_rate",
                    "reflection_rate",
                    "refinement_rate",
                    "avg_refinement_attempts",
                ],
                stage_rows,
            ),
            "",
        ]),
        encoding="utf-8",
    )

    if args.coverage_profile:
        coverage_payload = _write_coverage_artifacts(
            run_dir=run_dir,
            run_id=run_dir.name,
            rows=rows,
            profile=args.coverage_profile,
        )
        print(f"- run_coverage.md (missing={len(coverage_payload.get('missing', []))})")

    index_rows = _build_global_index(RESULTS_ROOT)
    print(f"Generated run reports in: {run_dir}")
    print(f"- {leaderboard_md.name}")
    print(f"- {failures_md.name}")
    print(f"- run_stage_diagnostics.md")
    print(f"Global index rows: {len(index_rows)} -> {INDEX_PATH}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = _read_jsonl(INDEX_PATH)
    if args.rebuild or not rows:
        rows = _build_global_index(RESULTS_ROOT)
    if not rows:
        print("No benchmark rows found.")
        return 0

    limit = max(1, args.limit)
    rows = rows[:limit]
    table_rows = []
    for r in rows:
        table_rows.append([
            r.get("run_id", ""),
            r.get("strategy", ""),
            r.get("corpus_tag", ""),
            r.get("reflection_token", ""),
            r.get("agentic_token", ""),
            r.get("status", ""),
            f"{_safe_float(r.get('avg_llm_judge_score', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_hallucination', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_answer_attempted', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_doc_match', 0.0)):.4f}",
            f"{_safe_float(r.get('avg_latency', 0.0)):.4f}",
            r.get("result_name", ""),
        ])
    print(_to_markdown_table(
        [
            "run_id",
            "strategy",
            "corpus_tag",
            "reflection",
            "agentic",
            "status",
            "avg_llm_judge",
            "avg_hallucination",
            "avg_answer_attempted",
            "avg_doc",
            "avg_latency",
            "file",
        ],
        table_rows,
    ))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    if not run_a.exists() or not run_b.exists():
        print("Both run directories must exist.")
        return 1

    rows_a = {}
    for fp, payload in _iter_primary_results(run_a):
        row = _result_row(run_a.name, fp, payload)
        rows_a[row["variant_id"]] = row

    rows_b = {}
    for fp, payload in _iter_primary_results(run_b):
        row = _result_row(run_b.name, fp, payload)
        rows_b[row["variant_id"]] = row

    keys = sorted(set(rows_a.keys()) | set(rows_b.keys()))
    if not keys:
        print("No comparable result rows found.")
        return 1

    out_rows = []
    for key in keys:
        a = rows_a.get(key)
        b = rows_b.get(key)
        a_score = _safe_float(a.get("avg_llm_judge_score", 0.0)) if a else 0.0
        b_score = _safe_float(b.get("avg_llm_judge_score", 0.0)) if b else 0.0
        a_hallu = _safe_float(a.get("avg_hallucination", 0.0)) if a else 0.0
        b_hallu = _safe_float(b.get("avg_hallucination", 0.0)) if b else 0.0
        a_answered = _safe_float(a.get("avg_answer_attempted", 0.0)) if a else 0.0
        b_answered = _safe_float(b.get("avg_answer_attempted", 0.0)) if b else 0.0
        a_doc = _safe_float(a.get("avg_doc_match", 0.0)) if a else 0.0
        b_doc = _safe_float(b.get("avg_doc_match", 0.0)) if b else 0.0
        a_lat = _safe_float(a.get("avg_latency", 0.0)) if a else 0.0
        b_lat = _safe_float(b.get("avg_latency", 0.0)) if b else 0.0
        strategy, corpus_tag, reflection, agentic = _split_variant_id(key)
        out_rows.append([
            strategy,
            corpus_tag,
            reflection,
            agentic,
            f"{a_score:.4f}",
            f"{b_score:.4f}",
            f"{(b_score - a_score):+.4f}",
            f"{a_hallu:.4f}",
            f"{b_hallu:.4f}",
            f"{(b_hallu - a_hallu):+.4f}",
            f"{a_answered:.4f}",
            f"{b_answered:.4f}",
            f"{(b_answered - a_answered):+.4f}",
            f"{a_doc:.4f}",
            f"{b_doc:.4f}",
            f"{(b_doc - a_doc):+.4f}",
            f"{a_lat:.4f}",
            f"{b_lat:.4f}",
            f"{(b_lat - a_lat):+.4f}",
        ])

    md = "\n".join([
        f"# Run Comparison: {run_a.name} vs {run_b.name}",
        "",
        _to_markdown_table(
            [
                "strategy",
                "corpus_tag",
                "reflection",
                "agentic",
                "A_judge",
                "B_judge",
                "delta_judge",
                "A_hallucination",
                "B_hallucination",
                "delta_hallucination",
                "A_answer_attempted",
                "B_answer_attempted",
                "delta_answer_attempted",
                "A_doc",
                "B_doc",
                "delta_doc",
                "A_latency",
                "B_latency",
                "delta_latency",
            ],
            out_rows,
        ),
        "",
    ])
    out_file = run_a / f"compare_{run_a.name}_vs_{run_b.name}.md"
    out_file.write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved: {out_file}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"run_dir not found: {run_dir}")
        return 1

    run_items = _iter_primary_results(run_dir)
    if not run_items:
        print(f"No benchmark result json found in {run_dir}")
        return 1

    rows = [_result_row(run_dir.name, fp, payload) for fp, payload in run_items]
    payload = _write_coverage_artifacts(
        run_dir=run_dir,
        run_id=run_dir.name,
        rows=rows,
        profile=args.profile,
    )
    print(f"Saved: {run_dir / 'run_coverage.md'}")
    print(
        "Coverage summary: "
        f"expected={payload.get('expected_count', 0)}, "
        f"present_unique={payload.get('present_unique_count', 0)}, "
        f"missing={len(payload.get('missing', []))}, "
        f"duplicates={len(payload.get('duplicates', []))}"
    )
    if args.strict and (payload.get("missing") or payload.get("duplicates")):
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark result reporting utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate run-level reports and global index")
    p_gen.add_argument("--run-dir", required=True, help="Run directory, e.g. data/results/20260218_111518")
    p_gen.add_argument("--top-k", type=int, default=50, help="Top-K failures to export")
    p_gen.add_argument(
        "--coverage-profile",
        default="",
        help="Also generate run_coverage.md using expected variant profile",
    )
    p_gen.set_defaults(func=cmd_generate)

    p_list = sub.add_parser("list", help="List recent benchmark rows from global index")
    p_list.add_argument("--limit", type=int, default=30)
    p_list.add_argument("--rebuild", action="store_true", help="Rebuild index by scanning data/results")
    p_list.set_defaults(func=cmd_list)

    p_cmp = sub.add_parser("compare", help="Compare two run directories")
    p_cmp.add_argument("--run-a", required=True)
    p_cmp.add_argument("--run-b", required=True)
    p_cmp.set_defaults(func=cmd_compare)

    p_cov = sub.add_parser("coverage", help="Check missing/duplicate variants for a run")
    p_cov.add_argument("--run-dir", required=True)
    p_cov.add_argument(
        "--profile",
        choices=["parallel_all", "parallel_all_agentic", "agentic_matrix"],
        default="parallel_all",
    )
    p_cov.add_argument("--strict", action="store_true", help="Return non-zero when missing/duplicate exists")
    p_cov.set_defaults(func=cmd_coverage)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
