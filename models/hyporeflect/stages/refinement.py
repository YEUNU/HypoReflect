"""Stage 5: Refinement (paper §3.2.5).

Two responsibilities:
1. RefinementHandler — single critique-conditional rewrite of the draft answer.
2. RefinementOrchestrator — outer R_max=2 loop with the lexicographic
   non-regression guard (paper §3.2.5):

       reflection_passed > grounded > not_insufficient > -issue_count
       > arithmetic_check > citation_present > single_answer_prefix

Hard guard: never replace a grounded answer with insufficient under
inapplicable_explain or zero_if_not_explicit policy. Forward guard: prefer
a substantive cited answer over a bare "insufficient evidence".
"""
import logging
import re
import time
from typing import Any, Optional

from core.config import RAGConfig
from utils.prompts import (
    FINAL_ANSWER_FORMAT_INSTRUCTION,
    REFINEMENT_RETRY_PROMPT,
    RESPONSE_REFINEMENT_PROMPT,
)
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import (
    _ANSWER_PREFIX_RE,
    CITATION_RE,
    extract_final_answer_from_json,
    missing_data_policy,
)
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class RefinementHandler:
    """Single critique-conditional rewrite (one pass)."""

    _MAX_JSON_ATTEMPTS = 3

    def __init__(self, llm, stage_model: str = ""):
        self.llm = llm
        self.stage_model = stage_model or RAGConfig.REFINEMENT_MODEL

    @staticmethod
    def _retry_message(failed_output: Any, reason: str) -> str:
        return REFINEMENT_RETRY_PROMPT.format(
            error=reason,
            previous_output=compact_json(failed_output, max_chars=900),
        )

    async def run(self, state: AgentState):
        started = time.perf_counter()
        base_messages = [
            {"role": "user", "content": RESPONSE_REFINEMENT_PROMPT.format(
                query=state.user_query,
                context=state.context,
                draft=state.final_answer,
                critique=state.critique,
                query_state=compact_json(state.query_state, max_chars=1200),
                evidence_ledger=compact_json({
                    "entries": state.evidence_ledger,
                    "missing_slots": state.missing_slots,
                }, max_chars=2000),
            )},
            {"role": "user", "content": FINAL_ANSWER_FORMAT_INSTRUCTION},
        ]

        def validate(data: dict[str, Any]) -> tuple[bool, str]:
            answer, reason = extract_final_answer_from_json(data)
            return bool(answer), reason

        def retry_message(data: dict[str, Any], reason: str) -> str:
            return self._retry_message(data, reason)

        data, ok, attempts = await generate_json_with_retries(
            self.llm,
            base_messages,
            validate,
            retry_message,
            max_attempts=self._MAX_JSON_ATTEMPTS,
            logger=logger,
            warning_prefix="refinement json generation failed",
            model=self.stage_model,
        )

        if ok:
            answer, _ = extract_final_answer_from_json(data)
            if answer:
                state.final_answer = answer
        append_trace(
            state.trace,
            step="refinement",
            input=base_messages,
            output={
                "final_answer": state.final_answer,
                "attempts": attempts,
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


class RefinementOrchestrator:
    """R_max=2 outer loop with lexicographic non-regression guard."""

    _ANSWER_PREFIX = "@@ANSWER:"

    def __init__(self, refinement: RefinementHandler, reflection, execution):
        self.refinement = refinement
        self.reflection = reflection
        self.execution = execution

    # ---------- gating helpers ----------
    @classmethod
    def _has_single_answer_prefix(cls, answer: str) -> bool:
        return cls._ANSWER_PREFIX in answer and len(_ANSWER_PREFIX_RE.findall(answer)) <= 1

    def _allow_no_citation(self, state: AgentState, answer: str) -> bool:
        return (
            missing_data_policy(state) == "zero_if_not_explicit"
            and bool(state.missing_slots)
            and self.execution._is_zero_policy_answer(answer)
        )

    def _has_required_citation(self, state: AgentState, answer: str) -> bool:
        if not RAGConfig.STRICT_CITATION_CHECK:
            return True
        if CITATION_RE.search(answer) is not None:
            return True
        return self._allow_no_citation(state, answer)

    def needs_refinement(self, state: AgentState) -> bool:
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

    # ---------- lexicographic ranking ----------
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

    # ---------- non-regression decision ----------
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
            if before_passed:
                return False, self._quality_gate_payload(
                    before_rank=before_rank,
                    after_rank=after_rank,
                    keep_after=False,
                    reason="non_regression_guard_before_grounded_after_insufficient",
                )
            if not after_passed and after_issues < before_issues:
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
        if before_insufficient and not after_insufficient:
            after_has_citation = CITATION_RE.search(after_text) is not None
            if after_has_citation:
                return True, self._quality_gate_payload(
                    before_rank=before_rank,
                    after_rank=after_rank,
                    keep_after=True,
                    reason="prefer_cited_over_insufficient",
                )
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

    # ---------- R_max loop entry ----------
    async def run_loop(self, state: AgentState, reflection_passed: bool) -> bool:
        prev_signature = self._refinement_signature(state)
        for _ in range(RAGConfig.MAX_REFINEMENT_ATTEMPTS):
            structural_needs_refinement = self.needs_refinement(state)
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
