import logging
import os
import re
from typing import Any, Optional

from core.config import RAGConfig
from core.vllm_client import get_llm_client
from models.hyporeflect.graphrag import GraphRAG
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import _ANSWER_PREFIX_RE, CITATION_RE, missing_data_policy
from models.hyporeflect.stages.perception import PerceptionHandler
from models.hyporeflect.stages.planning import PlanningHandler
from models.hyporeflect.stages.execution import ExecutionHandler
from models.hyporeflect.stages.reflection import ReflectionHandler
from models.hyporeflect.stages.refinement import RefinementHandler
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class AgentService:
    _ANSWER_PREFIX = "@@ANSWER:"
    _POST_REFINEMENT_OVERRIDE_BUILDERS = (
        ("operating_margin_driver_rule", "_build_operating_margin_driver_override_answer"),
        ("segment_drag_rule", "_build_segment_drag_override_answer"),
        ("debt_securities_listing_rule", "_build_debt_securities_override_answer"),
        ("quick_ratio_health_rule", "_build_quick_ratio_health_override_answer"),
        ("capital_intensity_ratio_rule", "_build_capital_intensity_override_answer"),
        ("dividend_stability_rule", "_build_dividend_stability_override_answer"),
    )

    def __init__(
        self,
        model_id: str = "local",
        strategy: str = "hyporeflect",
        corpus_tag: str = "default",
        llm_override: Any = None,
        grag_override: Any = None,
    ):
        self.llm = llm_override if llm_override is not None else get_llm_client(model_id)
        # Initialize with specific strategy for isolation unless an external retrieval backend is supplied.
        self.grag = grag_override if grag_override is not None else GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
        self.perception = PerceptionHandler(self.llm, stage_model=RAGConfig.PERCEPTION_MODEL)
        self.planning = PlanningHandler(self.llm, stage_model=RAGConfig.PLANNING_MODEL)
        self.execution = ExecutionHandler(self.llm, self.grag, stage_model=RAGConfig.EXECUTION_MODEL)
        self.reflection = ReflectionHandler(self.llm, stage_model=RAGConfig.REFLECTION_MODEL)
        self.refinement = RefinementHandler(self.llm, stage_model=RAGConfig.REFINEMENT_MODEL)

    def _allow_no_citation(self, state: AgentState, answer: str) -> bool:
        return (
            missing_data_policy(state) == "zero_if_not_explicit"
            and bool(state.missing_slots)
            and self.execution._is_zero_policy_answer(answer)
        )

    @classmethod
    def _has_single_answer_prefix(cls, answer: str) -> bool:
        return cls._ANSWER_PREFIX in answer and len(_ANSWER_PREFIX_RE.findall(answer)) <= 1

    def _has_required_citation(self, state: AgentState, answer: str) -> bool:
        if not RAGConfig.STRICT_CITATION_CHECK:
            return True
        if CITATION_RE.search(answer) is not None:
            return True
        return self._allow_no_citation(state, answer)

    def _needs_refinement(self, state: AgentState) -> bool:
        answer = str(state.final_answer or "")
        if not answer.strip():
            return True
        if not self._has_single_answer_prefix(answer):
            return True
        if self.execution._is_insufficient_answer(answer) and not state.missing_slots:
            return True
        if not self._has_required_citation(state, answer):
            return True
        return False

    @staticmethod
    def _arithmetic_rank(reflection_meta: Optional[dict[str, Any]]) -> int:
        arithmetic = str((reflection_meta or {}).get("arithmetic_check", "na") or "na").strip().lower()
        if arithmetic == "ok":
            return 2
        if arithmetic == "na":
            return 1
        return 0

    @staticmethod
    def _issue_count(reflection_meta: Optional[dict[str, Any]]) -> int:
        issues = (reflection_meta or {}).get("issues", [])
        if not isinstance(issues, list):
            return 0
        return len([str(item).strip() for item in issues if str(item).strip()])

    def _candidate_rank(
        self,
        state: AgentState,
        answer: str,
        reflection_passed: bool,
        reflection_meta: Optional[dict[str, Any]],
    ) -> tuple[int, int, int, int, int, int, int]:
        answer_text = str(answer or "")
        has_single_prefix = self._has_single_answer_prefix(answer_text)
        has_citation = self._has_required_citation(state, answer_text)
        issue_count = self._issue_count(reflection_meta)
        grounded, _ = self.execution._verify_answer_grounding(  # noqa: SLF001
            answer=answer_text,
            query_state=state.query_state,
            evidence_ledger=state.evidence_ledger,
            context=state.context,
            missing_slots=state.missing_slots,
        )
        return (
            int(bool(reflection_passed)),
            int(bool(grounded)),
            int(not self.execution._is_insufficient_answer(answer_text)),
            -issue_count,
            self._arithmetic_rank(reflection_meta),
            int(has_citation),
            int(has_single_prefix),
        )

    @staticmethod
    def _refinement_signature(state: AgentState) -> tuple[str, str, str, tuple[str, ...]]:
        answer = re.sub(r"\s+", " ", str(state.final_answer or "").strip().lower())
        critique = re.sub(r"\s+", " ", str(state.critique or "").strip().lower())
        meta = state.reflection_meta if isinstance(state.reflection_meta, dict) else {}
        arithmetic = str(meta.get("arithmetic_check", "na") or "na").strip().lower()
        issues_raw = meta.get("issues", [])
        issues: tuple[str, ...] = tuple(
            re.sub(r"\s+", " ", str(item).strip().lower())
            for item in (issues_raw if isinstance(issues_raw, list) else [])
            if str(item).strip()
        )
        return answer, critique, arithmetic, issues

    @staticmethod
    def _quality_gate_payload(
        before_rank: tuple[int, int, int, int, int, int, int],
        after_rank: tuple[int, int, int, int, int, int, int],
        keep_after: bool,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "before_rank": before_rank,
            "after_rank": after_rank,
            "decision": "keep_refined" if keep_after else "rollback_to_previous",
            "reason": reason,
        }

    def _prefer_refined_candidate(
        self,
        state: AgentState,
        before_answer: str,
        before_passed: bool,
        before_meta: Optional[dict[str, Any]],
        after_answer: str,
        after_passed: bool,
        after_meta: Optional[dict[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        before_text = str(before_answer or "")
        after_text = str(after_answer or "")
        before_insufficient = self.execution._is_insufficient_answer(before_text)
        after_insufficient = self.execution._is_insufficient_answer(after_text)
        before_rank = self._candidate_rank(state, before_answer, before_passed, before_meta)
        after_rank = self._candidate_rank(state, after_answer, after_passed, after_meta)

        # Hard non-regression guard: never replace a grounded answer with insufficient.
        if (not before_insufficient) and after_insufficient:
            policy = missing_data_policy(state)
            if policy in {"inapplicable_explain", "zero_if_not_explicit"}:
                return False, self._quality_gate_payload(
                    before_rank=before_rank,
                    after_rank=after_rank,
                    keep_after=False,
                    reason="policy_disallows_insufficient_after_grounded_answer",
                )
            before_issues = self._issue_count(before_meta)
            after_issues = self._issue_count(after_meta)
            # Exception: if both candidates fail reflection and insufficient candidate has
            # strictly fewer issues, allow rollback to avoid unsupported grounded claims.
            if before_passed:
                return False, self._quality_gate_payload(
                    before_rank=before_rank,
                    after_rank=after_rank,
                    keep_after=False,
                    reason="non_regression_guard_before_grounded_after_insufficient",
                )
            if not after_passed and after_issues < before_issues:
                # If both candidates fail reflection, prefer the lower-issue rollback
                # even when the previous answer carried a citation.
                return True, self._quality_gate_payload(
                    before_rank=before_rank,
                    after_rank=after_rank,
                    keep_after=True,
                    reason="issue_count_improved_on_insufficient_rollback",
                )
            return False, self._quality_gate_payload(
                before_rank=before_rank,
                after_rank=after_rank,
                keep_after=False,
                reason="non_regression_guard_before_grounded_after_insufficient",
            )

        # Forward guard: prefer a substantive answer over insufficient evidence.
        # This mirrors the non-regression guard in the opposite direction.
        # "insufficient evidence" always appears grounded=1 against an empty ledger,
        # which misleads the general rank comparison — handle explicitly here.
        if before_insufficient and not after_insufficient:
            after_has_citation = CITATION_RE.search(after_text) is not None
            if after_has_citation:
                # Cited answer: strong grounding signal, always prefer over insufficient.
                return True, self._quality_gate_payload(
                    before_rank=before_rank,
                    after_rank=after_rank,
                    keep_after=True,
                    reason="prefer_cited_over_insufficient",
                )
            # Uncited answer: compare from rank[2:] onward (skip the misleading
            # grounded field) so not_insufficient at [2] acts as tiebreaker.
            keep_after = after_rank[2:] > before_rank[2:]
            return keep_after, self._quality_gate_payload(
                before_rank=before_rank,
                after_rank=after_rank,
                keep_after=keep_after,
                reason="prefer_substantive_over_insufficient" if keep_after else "rank_comparison_no_grounded",
            )

        keep_after = after_rank > before_rank
        return keep_after, self._quality_gate_payload(
            before_rank=before_rank,
            after_rank=after_rank,
            keep_after=keep_after,
            reason="rank_comparison",
        )

    async def _run_refinement_loop(self, state: AgentState, reflection_passed: bool) -> bool:
        prev_signature = self._refinement_signature(state)
        for _ in range(RAGConfig.MAX_REFINEMENT_ATTEMPTS):
            structural_needs_refinement = self._needs_refinement(state)
            needs_refine = (not reflection_passed) or structural_needs_refinement
            append_trace(
                state.trace,
                step="refinement_gate",
                input={
                    "answer": state.final_answer,
                    "missing_slots": state.missing_slots,
                    "reflection_passed": reflection_passed,
                },
                output={
                    "needs_refinement": needs_refine,
                    "structural_needs_refinement": structural_needs_refinement,
                },
            )
            if not needs_refine:
                break

            before_answer = state.final_answer
            before_passed = reflection_passed
            before_critique = state.critique
            before_meta = dict(state.reflection_meta or {})

            await self.refinement.run(state)

            after_answer = state.final_answer
            after_passed = await self.reflection.run(state)
            after_meta = dict(state.reflection_meta or {})

            keep_refined, quality = self._prefer_refined_candidate(
                state=state,
                before_answer=before_answer,
                before_passed=before_passed,
                before_meta=before_meta,
                after_answer=after_answer,
                after_passed=after_passed,
                after_meta=after_meta,
            )
            append_trace(
                state.trace,
                step="refinement_quality_gate",
                input={
                    "before_answer": before_answer,
                    "before_passed": before_passed,
                    "before_meta": before_meta,
                    "after_answer": after_answer,
                    "after_passed": after_passed,
                    "after_meta": after_meta,
                },
                output=quality,
            )
            if keep_refined:
                reflection_passed = after_passed
            else:
                state.final_answer = before_answer
                state.critique = before_critique
                state.reflection_meta = before_meta
                reflection_passed = before_passed
            current_signature = self._refinement_signature(state)
            if current_signature == prev_signature:
                append_trace(
                    state.trace,
                    step="refinement_early_stop",
                    input={
                        "reflection_passed": reflection_passed,
                        "structural_needs_refinement": structural_needs_refinement,
                    },
                    output={
                        "reason": "no_state_change_after_refinement_attempt",
                    },
                )
                break
            prev_signature = current_signature
        return reflection_passed

    def _apply_post_refinement_overrides(self, state: AgentState) -> None:
        for reason, builder_name in self._POST_REFINEMENT_OVERRIDE_BUILDERS:
            builder = getattr(self.execution, builder_name, None)
            if builder is None:
                continue
            override_answer = builder(state)
            if not override_answer or override_answer == str(state.final_answer or ""):
                continue
            append_trace(
                state.trace,
                step="service_post_refinement_override",
                input={"before_answer": state.final_answer},
                output={
                    "after_answer": override_answer,
                    "reason": reason,
                },
            )
            state.final_answer = override_answer

    @classmethod
    def _ensure_answer_prefix(cls, answer: str) -> str:
        text = str(answer or "")
        if cls._ANSWER_PREFIX not in text:
            return f"{cls._ANSWER_PREFIX} {text}"
        return text

    @staticmethod
    def _build_simple_answer_prompt(context: str, user_query: str) -> str:
        return (
            "You are a financial QA assistant. Answer using only the provided context.\n"
            "Cite supporting evidence inline in this exact format: [[Title, Page X, Chunk Y]].\n"
            "If the context is insufficient, answer exactly: Insufficient evidence.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            "Answer:"
        )

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

    async def _run_non_agentic_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        """Single-pass retrieval + answer generation (no planning/reflection/refinement loop)."""
        _ = history
        context, nodes = await self.grag.retrieve(user_query, top_k=RAGConfig.DEFAULT_TOP_K)
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

    async def run_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        """
        Executes the Agentic Reasoning Loop (Perception, Planning, Execution, Reflection, Refinement).
        """
        agentic_mode = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
        if agentic_mode == "off":
            logger.info("HypoReflect running in agentic off mode.")
            return await self._run_non_agentic_workflow(user_query, history)

        state = AgentState(user_query, history or [])
        # Perception -> Planning -> Execution
        await self.perception.run(state)
        await self.planning.run(state)
        await self.execution.run(state)

        # Reflection audit + deterministic refinement gate
        if RAGConfig.ENABLE_AGENT_REFLECTION:
            reflection_passed = await self.reflection.run(state)
            await self._run_refinement_loop(state, reflection_passed)
        else:
            logger.info("Ablation: Skipping Reflection & Refinement stages.")

        self._apply_post_refinement_overrides(state)

        # Ensure final answer formatting with mandatory prefix
        state.final_answer = self._ensure_answer_prefix(state.final_answer)

        # Format sources for metric evaluation: {"doc": title, "page": page, "text": text}
        unique_sources = self._build_unique_sources(state.all_context_data)
        return state.final_answer, unique_sources, state.trace
