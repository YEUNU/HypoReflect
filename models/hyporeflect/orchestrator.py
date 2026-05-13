"""Five-stage reflective agent orchestrator (paper §3.2).

Sequences Perception -> Planning -> Execution -> Reflection -> Refinement.
Branches into a single-pass non-agentic path when RAG_AGENTIC_MODE=off (the
agentic-OFF retrieval baseline used in paper §4.4 ablation table).

Responsibilities split out from AgentService:
- Refinement ranking + R_max loop -> stages/refinement.RefinementOrchestrator
- Non-agentic prompt building -> _build_simple_answer_prompt
"""
import logging
import os
import re
from typing import Any, Optional

from core.config import RAGConfig
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.refinement import RefinementOrchestrator
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


_ANSWER_PREFIX = "@@ANSWER:"


class Orchestrator:
    """Sequences the five-stage agentic pipeline; non-agentic baseline lives
    on the same class so the entry point matches paper §4.4 settings."""

    def __init__(
        self,
        *,
        llm,
        grag,
        perception,
        planning,
        execution,
        reflection,
        refinement,
    ):
        self.llm = llm
        self.grag = grag
        self.perception = perception
        self.planning = planning
        self.execution = execution
        self.reflection = reflection
        self.refinement = refinement
        self.refinement_loop = RefinementOrchestrator(
            refinement=refinement,
            reflection=reflection,
            execution=execution,
        )

    # ---------- formatting ----------
    @classmethod
    def _ensure_answer_prefix(cls, answer: str) -> str:
        text = str(answer or "")
        if _ANSWER_PREFIX not in text:
            return f"{_ANSWER_PREFIX} {text}"
        return text

    @staticmethod
    def _strip_format_instruction(query: str) -> str:
        """Remove the benchmark-format suffix that cli/benchmark.py may attach
        before sending the query to retrieval. The format instruction is
        answer-formatting noise when embedded for ANN search."""
        marker = "[Benchmark Output Format]"
        if marker in query:
            return query.split(marker, 1)[0].strip()
        return query

    @staticmethod
    def _build_unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_sources: list[dict[str, Any]] = []
        seen = set()
        for row in rows:
            doc = row.get("title") or row.get("doc") or "Unknown"
            page = row.get("page", 0)
            sent_id = row.get("sent_id", 0)
            key = (doc, page, sent_id)
            if key in seen:
                continue
            unique_sources.append(
                {
                    "doc": doc,
                    "page": page,
                    "text": row.get("text", ""),
                    "sent_id": sent_id,
                }
            )
            seen.add(key)
        return unique_sources

    # ---------- non-agentic baseline ----------
    @staticmethod
    def _build_simple_answer_prompt(context: str, user_query: str) -> str:
        # Agentic-OFF retrieval baseline (paper §4.4). Identical structure to
        # the HopRAG / naive baseline prompts so any score gap traces back to
        # retrieval, not synthesis-prompt asymmetry. The `_ensure_answer_prefix`
        # wrapper attaches `@@ANSWER:` downstream; doc_match/page_match come
        # from retrieved nodes, not the answer text.
        return (
            "You are a financial analyst. Answer the question using only the provided context.\n"
            "If the context is insufficient, say you do not know.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question: {user_query}\n"
            "\n"
            "Answer:"
        )

    async def _run_non_agentic_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        _ = history
        retrieval_query = self._strip_format_instruction(user_query)
        graph_depth = RAGConfig.AGENTIC_OFF_GRAPH_DEPTH
        if graph_depth > 0:
            # Paper §3.2.3 graph traversal in deterministic mode: no LLM
            # continuation check (force_expand=True). Activates NEXT/HOP
            # edges + runtime-HOP that were previously dead weight in
            # agentic-off (only the agentic-on graph_search tool used them).
            context, nodes = await self.grag.graph_search(
                entities=[retrieval_query],
                depth=graph_depth,
                top_k=RAGConfig.DEFAULT_TOP_K,
                user_query=retrieval_query,
                force_expand=True,
            )
        else:
            # Legacy path (Stage 1+2 RRF + rerank only) — kept for ablation
            # under RAG_AGENTIC_OFF_GRAPH_DEPTH=0.
            context, nodes = await self.grag.retrieve(retrieval_query, top_k=RAGConfig.DEFAULT_TOP_K)
        retrieved_nodes = nodes if isinstance(nodes, list) else []
        unique_sources = self._build_unique_sources(retrieved_nodes)

        trace: list[dict[str, Any]] = [
            {
                "step": "agentic_off_retrieve",
                "input": {"query": user_query, "top_k": RAGConfig.DEFAULT_TOP_K},
                "output": {"retrieved_sources": len(unique_sources)},
            }
        ]

        if not context:
            answer = self._ensure_answer_prefix("Insufficient evidence.")
            trace.append(
                {
                    "step": "agentic_off_answer",
                    "input": {"query": user_query},
                    "output": {"answer": answer, "reason": "empty_context"},
                }
            )
            return answer, unique_sources, trace

        prompt = self._build_simple_answer_prompt(context, user_query)
        messages = [{"role": "user", "content": prompt}]
        answer = await self.llm.generate_response(messages)
        answer = self._ensure_answer_prefix(str(answer or ""))
        trace.append(
            {
                "step": "agentic_off_answer",
                "input": {"query": user_query},
                "output": {"answer": answer},
            }
        )
        return answer, unique_sources, trace

    # ---------- main entry ----------
    async def run_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        agentic_mode = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
        if agentic_mode == "off":
            logger.info("HypoReflect running in agentic off mode.")
            return await self._run_non_agentic_workflow(user_query, history)

        state = AgentState(user_query, history or [])
        stage_models = {
            "perception": self.perception.effective_model(),
            "planning": self.planning.effective_model(),
            "execution": self.execution.effective_model(),
            "reflection": self.reflection.effective_model(),
            "refinement": self.refinement.effective_model(),
            "default_model": RAGConfig.DEFAULT_MODEL,
            "agentic_mode": agentic_mode or "on",
        }
        logger.info("HypoReflect stage models: %s", stage_models)
        append_trace(state.trace, step="stage_models", output=stage_models)
        await self.perception.run(state)
        await self.planning.run(state)
        await self.execution.run(state)

        if RAGConfig.ENABLE_AGENT_REFLECTION:
            reflection_passed = await self.reflection.run(state)
            await self.refinement_loop.run_loop(state, reflection_passed)
        else:
            logger.info("Ablation: Skipping Reflection & Refinement stages.")

        await self._post_pipeline_validator(state)

        state.final_answer = self._ensure_answer_prefix(state.final_answer)

        unique_sources = self._build_unique_sources(state.all_context_data)
        return state.final_answer, unique_sources, state.trace

    _VALIDATOR_PROMPT = (
        "You verify a financial QA answer.\n"
        "QUERY_STATE specifies the exact entity, period, and metric the\n"
        "question targets. ANSWER must match all three. Treat standard\n"
        "line-item synonyms as equivalent (capex ≡ purchases of PP&E;\n"
        "revenue ≡ net sales; cost of revenue ≡ cost of sales; net income\n"
        "≡ net earnings; operating income ≡ income from operations).\n"
        "Reject ANSWER (=NO) if any of these hold:\n"
        "- it reports a value for a different metric than QUERY_STATE.metric\n"
        "  (e.g., long-term debt only when the question asks total debt,\n"
        "   one facility amount when the question asks the total across\n"
        "   facilities, a country list when the question asks regional\n"
        "   segmentation, or vice versa);\n"
        "- the reported value's period or entity differs from QUERY_STATE;\n"
        "- the cited chunk's text contains the value but the value is\n"
        "  associated there with a different metric/line-item than what\n"
        "  ANSWER claims;\n"
        "- the question asked for an explanation/conditional response and\n"
        "  ANSWER is a bare yes/no with no supporting reasoning.\n"
        "Accept ANSWER (=YES) otherwise, including when ANSWER is the\n"
        "literal `@@ANSWER: insufficient evidence` (an honest abstain).\n"
        "\n"
        "QUERY: {query}\n"
        "QUERY_STATE: {query_state}\n"
        "EVIDENCE_LEDGER: {evidence_ledger}\n"
        "CONTEXT: {context}\n"
        "ANSWER: {final_answer}\n"
        "\n"
        "Reply with a single JSON object: {{\"verdict\": \"yes\"|\"no\", \"reason\": \"...\"}}"
    )

    _NUM_TOKEN_RE = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|trillion|m|b|%)?", re.IGNORECASE)
    _CITATION_RE = re.compile(r"\[\[([^\]]+?),\s*Page\s*(\d+)\s*,\s*Chunk\s*(\d+)\s*\]\]", re.IGNORECASE)

    @staticmethod
    def _normalize_num(token: str) -> str:
        t = token.lower().strip()
        t = t.replace("$", "").replace(",", "").replace(" ", "")
        for suffix in ("million", "billion", "trillion", "%"):
            if t.endswith(suffix):
                t = t[: -len(suffix)]
        if t.endswith("m") and len(t) > 1 and t[-2].isdigit():
            t = t[:-1]
        elif t.endswith("b") and len(t) > 1 and t[-2].isdigit():
            t = t[:-1]
        return t.rstrip(".")

    async def _post_pipeline_validator(self, state: AgentState) -> None:
        """Deterministic verbatim-value gate (lever 6).

        For each numeric token in the answer text, verify that the same
        numeric value appears in at least one cited chunk's text after
        common normalisation (strip $, commas, whitespace, unit suffix).
        If ANY numeric token in the answer cannot be located in any cited
        chunk and the answer is not exempt (compute query whose result is
        derived), force `@@ANSWER: insufficient evidence`.

        No LLM call — purely structural. Skips named-entity-only answers
        (no numeric tokens) and compute-query final results. Reduces false
        positives observed under the prior LLM-based validator.
        """
        from models.hyporeflect.trace import append_trace as _append

        answer_text = str(state.final_answer or "")
        if "insufficient evidence" in answer_text.lower():
            return
        if not answer_text.strip():
            return

        # Compute-query final results are derived and exempt from the
        # verbatim check (e.g., 30.8% computed from operands in chunks).
        # Operand values themselves still need to appear in chunks but
        # those are normally extra mentions in the answer text near the
        # result — they pass naturally when present, fail when absent.
        answer_type = str((state.query_state or {}).get("answer_type", "") or "").strip().lower()

        nums_in_answer = self._NUM_TOKEN_RE.findall(answer_text)
        # Strip the citation chunk/page numbers from the candidate list
        # (e.g., `Page 41` inside `[[3M_2018_10K, Page 41, Chunk 374]]`).
        citation_spans = list(self._CITATION_RE.finditer(answer_text))
        citation_char_ranges = [(m.start(), m.end()) for m in citation_spans]
        def _in_citation_span(needle: str) -> bool:
            idx = answer_text.find(needle)
            while idx != -1:
                in_span = any(s <= idx < e for s, e in citation_char_ranges)
                if not in_span:
                    return False
                idx = answer_text.find(needle, idx + 1)
            return True

        candidate_values: list[str] = []
        for token in nums_in_answer:
            token_str = token.strip()
            if not token_str:
                continue
            if _in_citation_span(token_str):
                continue
            normalized = self._normalize_num(token_str)
            if not normalized or normalized in {"0", "0.0"}:
                continue
            # Skip very short integers (years, single digits) — too noisy.
            digits_only = normalized.lstrip("-")
            if "." not in digits_only and len(digits_only) <= 1:
                continue
            candidate_values.append(normalized)

        if not candidate_values:
            _append(state.trace, step="post_pipeline_validator",
                    output={"verdict": "skip", "reason": "no numeric values to verify"})
            return

        # Gather text of cited chunks from all_context_data.
        cited_keys: set[tuple[str, int, int]] = set()
        for m in citation_spans:
            try:
                title = m.group(1).strip()
                page = int(m.group(2))
                chunk = int(m.group(3))
                cited_keys.add((title, page, chunk))
            except Exception:
                continue
        chunk_texts: list[str] = []
        for node in (state.all_context_data or []):
            if not isinstance(node, dict):
                continue
            key = (
                str(node.get("title") or node.get("doc") or "").strip(),
                int(node.get("page", 0) or 0),
                int(node.get("sent_id", 0) or 0),
            )
            if key in cited_keys:
                chunk_texts.append(str(node.get("text", "") or ""))
        if not chunk_texts:
            # No cited chunks recoverable — can't validate, leave answer.
            _append(state.trace, step="post_pipeline_validator",
                    output={"verdict": "skip", "reason": "cited chunks not found in pool"})
            return

        combined_norm = " ".join(
            self._normalize_num(t)
            for chunk_text in chunk_texts
            for t in self._NUM_TOKEN_RE.findall(chunk_text)
        )
        missing: list[str] = [v for v in candidate_values if v not in combined_norm]

        # If this is a compute query, the FIRST distinct missing value
        # (typically the final result, e.g., 30.8%) is exempt — only flag
        # when additional asserted operand values are also missing.
        if answer_type == "compute" and missing:
            seen = set()
            unique_missing = []
            for v in missing:
                if v not in seen:
                    seen.add(v)
                    unique_missing.append(v)
            if len(unique_missing) <= 1:
                _append(state.trace, step="post_pipeline_validator",
                        output={"verdict": "skip", "reason": "compute result exempt",
                                "missing": unique_missing})
                return
            missing = unique_missing[1:]

        if missing:
            replaced_from = answer_text
            state.final_answer = "@@ANSWER: insufficient evidence"
            _append(state.trace, step="post_pipeline_validator",
                    output={"verdict": "no",
                            "reason": f"numeric values not in cited chunks: {missing[:4]}",
                            "replaced_from": replaced_from[:200],
                            "final_answer": state.final_answer})
        else:
            _append(state.trace, step="post_pipeline_validator",
                    output={"verdict": "yes", "values_checked": len(candidate_values)})
