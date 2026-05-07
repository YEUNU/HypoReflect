import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from core.config import RAGConfig
from core.vllm_client import get_llm_client
from models.hyporeflect.service import AgentService
from models.naive.naive_rag import NaiveRAG
from utils.io import _safe_float
from utils.metrics import evaluate_financebench_response
from utils.prompts import BENCHMARK_MATH_FORMAT_INSTRUCTION, BENCHMARK_MCQ_JSON_FORMAT_INSTRUCTION
from utils.reporting import _write_model_report_artifacts


logger = logging.getLogger("HypoReflect")


def _as_lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_LABEL_RE = re.compile(
    r"(?is)(?:final\s+answer|@@ANSWER|answer)\s*:?\s*(.+?)(?:\n\n|\Z)"
)


def _extract_final_answer(answer_text: str) -> str:
    """Extract the final answer from a model response that may contain
    step-by-step reasoning. Order: \\boxed{...} > 'Final Answer:' marker >
    last 300 chars. Avoids substring-matching the reasoning body for
    abstain detection.
    """
    if not answer_text:
        return ""
    boxed = _BOXED_RE.findall(answer_text)
    if boxed:
        return boxed[-1].strip()
    matches = _FINAL_LABEL_RE.findall(answer_text)
    if matches:
        return matches[-1].strip()[:400]
    return answer_text[-300:].strip()


def _is_multiple_choice_query(item: dict[str, Any]) -> bool:
    for key in ("choices", "options", "answer_choices"):
        value = item.get(key)
        if isinstance(value, dict) and len(value) >= 2:
            return True
        if isinstance(value, list) and len(value) >= 2:
            return True

    format_blob = " ".join(
        _as_lower_text(item.get(key))
        for key in ("answer_type", "question_type", "question_format", "task_type")
    )
    if any(tag in format_blob for tag in ("mcq", "multiple choice", "multiple-choice")):
        return True

    query = str(item.get("query", "") or "")
    alpha_choices = re.findall(r"(?m)^\s*([A-Ha-h])[\).]\s+", query)
    paren_choices = re.findall(r"(?m)[\(\[]([A-Ha-h])[\)\]]\s+", query)
    return len(set(letter.upper() for letter in alpha_choices + paren_choices)) >= 2


def _is_math_query(item: dict[str, Any]) -> bool:
    answer_type = _as_lower_text(item.get("answer_type"))
    if answer_type in {"compute", "math", "numerical", "calculation"}:
        return True

    reasoning_blob = " ".join(
        _as_lower_text(item.get(key))
        for key in ("question_reasoning", "question_type", "reasoning_type", "task_type")
    )
    if any(
        token in reasoning_blob
        for token in ("numerical reasoning", "math", "mathematical", "calculation", "compute", "arithmetic")
    ):
        return True

    query = _as_lower_text(item.get("query"))
    legacy_math_pattern = r"\b(calculate|compute|ratio|percentage|percent change|difference|average)\b"
    return bool(re.search(legacy_math_pattern, query)) and bool(re.search(r"\d", query))


def _build_benchmark_query(query: str, item: dict[str, Any]) -> str:
    """Return the user-facing query as-is.

    The previous implementation appended `[Benchmark Output Format]` blocks
    instructing the model to produce step-by-step CoT inside `\\boxed{}`. That
    suffix (a) leaked into retrieval embeddings as noise, (b) forced verbose
    reasoning that doubled latency, and (c) collided with the citation-first
    answer format expected by `simple_answer_prompt`. The judge prompt now
    handles `\\boxed{}` / "Final Answer:" extraction internally, so emitting
    that scaffolding upstream provides no signal — only confounds.

    Math/MCQ format instructions are kept available as constants but are not
    injected into the live benchmark query. Re-enable behind a config flag
    if a future evaluation legitimately needs them.
    """
    _ = item  # kept for signature stability; type detection no longer alters the query.
    return query


async def run_benchmark(
    queries_file: str,
    strategy: str,
    model_id: str,
    is_batch: bool = False,
    sample_companies: Optional[list[str]] = None,
    corpus_tag: str = "default",
    output_dir: Optional[Path] = None,
    agentic_mode: Optional[str] = None,
    limit: Optional[int] = None,
):
    normalized_agentic: Optional[str] = None
    if agentic_mode is not None:
        candidate = str(agentic_mode).strip().lower()
        if candidate in {"on", "off"}:
            normalized_agentic = candidate
            os.environ["RAG_AGENTIC_MODE"] = candidate

    env_agentic = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
    effective_agentic: Optional[str] = env_agentic if env_agentic in {"on", "off"} else normalized_agentic

    try:
        if strategy == "hyporeflect":
            engine = AgentService(model_id=model_id, strategy=strategy, corpus_tag=corpus_tag)
        elif strategy == "naive":
            engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
        elif strategy == "hoprag":
            from models.hoprag.hoprag_adapter import HopRAGAdapter

            engine = HopRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
        elif strategy == "ms_graphrag":
            from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter

            engine = MSGraphRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
        else:
            logger.error("Unknown strategy: %s", strategy)
            return None

        vllm = get_llm_client(model_id)
    except Exception as exc:
        logger.error("Failed to initialize engine for %s: %s", strategy, exc)
        return None

    if not os.path.exists(queries_file):
        logger.error("Queries file %s not found.", queries_file)
        return None

    with open(queries_file, "r", encoding="utf-8") as file:
        benchmark_data = json.load(file)

    if sample_companies:
        initial_len = len(benchmark_data)
        benchmark_data = [item for item in benchmark_data if item.get("company") in sample_companies]
        logger.info(
            "Filtering for %d sample companies: %d -> %d queries",
            len(sample_companies),
            initial_len,
            len(benchmark_data),
        )

    if limit is not None:
        benchmark_data = benchmark_data[: max(0, int(limit))]
        logger.info("--limit %d: evaluating %d queries", limit, len(benchmark_data))

    dataset_marker = benchmark_data[0].get("dataset", "") if benchmark_data else ""
    is_financebench = dataset_marker == "financebench"
    dataset_name = "FinanceBench" if is_financebench else "Unknown"
    results = []
    category_results = {}

    agentic_log = effective_agentic if effective_agentic else "native"
    logger.info(
        "Starting benchmark [%s] on %s | Queries: %d | Agentic: %s",
        strategy,
        dataset_name,
        len(benchmark_data),
        agentic_log,
    )

    if output_dir:
        results_dir = output_dir
    else:
        env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
        start_timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("data/results") / start_timestamp

    results_dir.mkdir(parents=True, exist_ok=True)
    model_results_dir = results_dir / strategy
    model_results_dir.mkdir(parents=True, exist_ok=True)
    ablation_results_dir = model_results_dir / corpus_tag
    ablation_results_dir.mkdir(parents=True, exist_ok=True)

    is_hyporeflect = strategy.lower() == "hyporeflect"
    refl_suffix = "_no_refl" if is_hyporeflect and not RAGConfig.ENABLE_AGENT_REFLECTION else ""
    sample_suffix = "_sample" if sample_companies else ""
    agentic_suffix = ""
    output_results_dir = ablation_results_dir

    if is_hyporeflect:
        reflection_dirname = "refl_on" if RAGConfig.ENABLE_AGENT_REFLECTION else "refl_off"
        output_results_dir = output_results_dir / reflection_dirname
    if effective_agentic:
        output_results_dir = output_results_dir / f"agentic_{effective_agentic}"
        agentic_suffix = f"_agentic_{effective_agentic}"
    output_results_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_results_dir / f"{strategy}_{corpus_tag}{refl_suffix}{agentic_suffix}{sample_suffix}.json"
    summary: dict[str, Any] = {}

    for idx, item in enumerate(benchmark_data):
        original_query = item["query"]
        query = _build_benchmark_query(original_query, item)
        ground_truth = item.get("ground_truth", "")
        category = item.get("category", "Uncategorized")

        started = time.time()
        try:
            response, retrieved_sources, trace = await engine.run_workflow(query, [])
            latency = time.time() - started

            if is_financebench:
                metrics = await evaluate_financebench_response(
                    query=original_query,
                    response=response,
                    ground_truth=ground_truth,
                    retrieved_sources=retrieved_sources,
                    expected_doc=item.get("evidence_doc", ""),
                    expected_page=item.get("evidence_page"),
                    vllm_client=vllm,
                )
                result_item = {
                    "query": original_query,
                    "category": category,
                    "answer": response,
                    "ground_truth": ground_truth,
                    "expected_sources": {
                        "doc": item.get("evidence_doc", ""),
                        "page": item.get("evidence_page"),
                        "text": item.get("evidence_text", ""),
                    },
                    "retrieved_sources": retrieved_sources,
                    "interaction_trace": trace,
                    "latency": latency,
                    **metrics,
                }
        except Exception as exc:
            logger.error("Error processing query '%s': %s", original_query, exc)
            import traceback

            logger.error(traceback.format_exc())
            latency = time.time() - started
            error_text = f"{type(exc).__name__}: {exc}"

            if is_financebench:
                metrics = {
                    "llm_judge_score": 0.0,
                    "llm_judge_reason": "runtime_error",
                    "hallucination": 0.0,
                    "hallucination_reason": "runtime_error",
                    "hallucination_source": "runtime_error",
                    "hallucination_model": str(RAGConfig.HALLUCINATION_EVAL_MODEL or ""),
                    "doc_match": 0.0,
                    "page_match": 0.0,
                }
                result_item = {
                    "query": original_query,
                    "category": category,
                    "answer": f"@@ANSWER: ERROR - {error_text}",
                    "ground_truth": ground_truth,
                    "expected_sources": {
                        "doc": item.get("evidence_doc", ""),
                        "page": item.get("evidence_page"),
                        "text": item.get("evidence_text", ""),
                    },
                    "retrieved_sources": [],
                    "interaction_trace": [{"step": "error", "output": error_text}],
                    "latency": latency,
                    "error": error_text,
                    **metrics,
                }

        if query != original_query:
            result_item["benchmark_query"] = query

        if is_financebench:
            answer_text = str(result_item.get("answer", "") or "")
            has_error = bool(result_item.get("error"))
            # Detect abstain on the EXTRACTED final answer (\\boxed{} or
            # 'Final Answer:' marker), NOT on the full reasoning body. Step-by-
            # step CoT often uses 'insufficient evidence' as a logical token
            # while still arriving at a substantive answer; substring matching
            # the full text mis-classifies those as abstains.
            final_answer = _extract_final_answer(answer_text).lower()
            is_abstain = "insufficient evidence" in final_answer
            judge_score = _safe_float(result_item.get("llm_judge_score", 0.0), 0.0)
            # Judge override: if the LLM judge already scored this as correct,
            # the model clearly produced a usable answer regardless of phrasing.
            if has_error:
                answer_attempted = 0.0
            elif judge_score >= 0.5:
                answer_attempted = 1.0
            else:
                answer_attempted = 0.0 if is_abstain else 1.0
            result_item["answer_attempted"] = answer_attempted
            result_item["final_answer_extracted"] = final_answer[:300]
            if not isinstance(result_item.get("hallucination"), (int, float)):
                result_item["hallucination"] = 1.0 if (answer_attempted > 0.0 and judge_score < 1.0) else 0.0

        results.append(result_item)
        if category not in category_results:
            category_results[category] = []
        category_results[category].append(result_item)

        error_suffix = " [ERROR]" if result_item.get("error") else ""
        print(
            f"[{strategy}] ({idx + 1}/{len(benchmark_data)}) [{category}]{error_suffix} "
            f"Judge: {metrics['llm_judge_score']:.1f} | Hallu: {result_item.get('hallucination', 0.0):.0f} "
            f"| DocMatch: {metrics['doc_match']:.0f} | Latency: {latency:.1f}s"
        )

        summary = {
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "dataset": dataset_name,
            "queries_count": len(results),
            "total_queries": len(benchmark_data),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "in_progress" if len(results) < len(benchmark_data) else "completed",
            "ablation": {
                "table_to_text": RAGConfig.ABLATION_TABLE_TO_TEXT,
                "adaptive_chunking": RAGConfig.ABLATION_ADAPTIVE_CHUNKING,
                "rolling_summary": RAGConfig.ABLATION_ROLLING_SUMMARY,
                "enable_reflection": RAGConfig.ENABLE_AGENT_REFLECTION,
            },
        }
        if effective_agentic:
            summary["agentic_mode"] = effective_agentic
        for key in result_item.keys():
            if isinstance(result_item[key], (int, float)):
                summary[f"avg_{key}"] = sum(result[key] for result in results) / len(results)

        cat_summaries = {}
        for cat, cat_list in category_results.items():
            cat_sum = {"count": len(cat_list)}
            for key in result_item.keys():
                if isinstance(result_item[key], (int, float)):
                    cat_sum[f"avg_{key}"] = sum(result[key] for result in cat_list) / len(cat_list)
            cat_summaries[cat] = cat_sum
        summary["category_summaries"] = cat_summaries
        summary["details"] = results

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        try:
            _write_model_report_artifacts(summary, result_file)
        except Exception as exc:
            logger.warning("Failed to write report artifacts for %s: %s", result_file, exc)

    if not results:
        return None

    def _make_gate_check(actual: float, target: float, mode: str) -> dict[str, Any]:
        passed = actual >= target if mode == "min" else actual <= target
        return {"mode": mode, "target": target, "actual": actual, "passed": passed}

    gate_payload: dict[str, Any] = {"enabled": RAGConfig.BENCHMARK_GATE_ENABLED, "passed": None, "checks": {}}
    if RAGConfig.BENCHMARK_GATE_ENABLED:
        checks: dict[str, dict[str, Any]] = {}
        avg_latency = float(summary.get("avg_latency", 0.0))
        checks["avg_latency"] = _make_gate_check(avg_latency, RAGConfig.BENCHMARK_MAX_AVG_LATENCY, "max")
        checks["avg_llm_judge_score"] = _make_gate_check(
            float(summary.get("avg_llm_judge_score", 0.0)),
            RAGConfig.BENCHMARK_MIN_LLM_JUDGE,
            "min",
        )
        checks["avg_doc_match"] = _make_gate_check(
            float(summary.get("avg_doc_match", 0.0)),
            RAGConfig.BENCHMARK_MIN_DOC_MATCH,
            "min",
        )

        gate_payload["checks"] = checks
        gate_payload["passed"] = all(item.get("passed", False) for item in checks.values())
    summary["benchmark_gate"] = gate_payload

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    try:
        _write_model_report_artifacts(summary, result_file)
    except Exception as exc:
        logger.warning("Failed to write final report artifacts for %s: %s", result_file, exc)

    print(f"\n{'=' * 50}")
    print(f"[{strategy.upper()}] Benchmark Complete - {dataset_name}")
    print(f"{'=' * 50}")
    for key, value in summary.items():
        if key.startswith("avg_"):
            print(f"  Overall {key}: {value:.4f}")

    print("\nCategory Breakdown:")
    for cat, cat_sum in summary["category_summaries"].items():
        print(f"  - {cat} (n={cat_sum['count']}):")
        for key, value in cat_sum.items():
            if key.startswith("avg_"):
                print(f"    {key}: {value:.4f}")

    if summary["benchmark_gate"]["enabled"]:
        gate_result = "PASS" if summary["benchmark_gate"]["passed"] else "FAIL"
        print(f"\nBenchmark Gate: {gate_result}")
        for name, check in summary["benchmark_gate"]["checks"].items():
            target_str = f">= {check['target']:.4f}" if check["mode"] == "min" else f"<= {check['target']:.4f}"
            print(f"  {name}: {check['actual']:.4f} (target {target_str}) -> {'PASS' if check['passed'] else 'FAIL'}")

    print(f"\n  Final results saved to: {result_file}")
    print(f"{'=' * 50}\n")
    return summary
