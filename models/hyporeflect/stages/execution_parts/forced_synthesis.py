import time
from typing import Any, Optional

from models.hyporeflect.state import AgentState
from models.hyporeflect.trace import append_trace
from utils.prompts import CHAT_SYSTEM_FORMAT_INSTRUCTION, FINAL_ANSWER_FORMAT_INSTRUCTION


class ForcedSynthesisSupport:
    def _run_compute_slot_fill_before_synthesis(self, state: AgentState) -> None:
        missing_before = list(state.missing_slots)
        fill_entries = self._deterministic_compute_slot_entries(
            query_state=state.query_state,
            missing_slots=state.missing_slots,
            nodes=self._dedupe_nodes(state.all_context_data, max_nodes=28),
        )
        if not fill_entries:
            return
        ledger_before = len(state.evidence_ledger)
        state.evidence_ledger = self._merge_ledger(state.evidence_ledger, fill_entries)
        state.missing_slots = self._resolve_missing_slots(
            state.query_state,
            state.evidence_ledger,
            model_missing_slots=None,
            trust_model_missing=False,
        )
        append_trace(
            state.trace,
            step="execution_compute_slot_fill",
            input={
                "missing_slots_before": missing_before,
            },
            output={
                "new_entries": len(fill_entries),
                "ledger_size_before": ledger_before,
                "ledger_size_after": len(state.evidence_ledger),
                "missing_slots_after": state.missing_slots,
            },
        )

    def _run_compute_slot_realign_before_synthesis(self, state: AgentState) -> None:
        collapsed_slots_before = self._collapsed_multi_period_slots(
            state.query_state,
            state.evidence_ledger,
        )
        if not collapsed_slots_before:
            return

        realign_entries = self._deterministic_compute_slot_entries(
            query_state=state.query_state,
            missing_slots=collapsed_slots_before,
            nodes=self._dedupe_nodes(state.all_context_data, max_nodes=28),
        )
        applied = False
        collapsed_slots_after: list[Any] = collapsed_slots_before
        if realign_entries:
            candidate_ledger = self._merge_ledger(state.evidence_ledger, realign_entries)
            candidate_missing = self._resolve_missing_slots(
                state.query_state,
                candidate_ledger,
                model_missing_slots=None,
                trust_model_missing=False,
            )
            collapsed_slots_after = self._collapsed_multi_period_slots(
                state.query_state,
                candidate_ledger,
            )
            if (
                not candidate_missing
                and len(collapsed_slots_after) < len(collapsed_slots_before)
            ):
                state.evidence_ledger = candidate_ledger
                state.missing_slots = candidate_missing
                applied = True
        append_trace(
            state.trace,
            step="execution_compute_slot_realign",
            input={
                "collapsed_slots_before": collapsed_slots_before,
            },
            output={
                "realign_entries": len(realign_entries),
                "applied": applied,
                "collapsed_slots_after": collapsed_slots_after,
                "missing_slots_after": state.missing_slots,
            },
        )

    async def _run_compute_calculator_before_synthesis(
        self,
        state: AgentState,
        answer_type: str,
    ) -> tuple[Optional[dict[str, Any]], str]:
        calc: Optional[dict[str, Any]] = None
        calc_hint = ""
        if answer_type != "compute" or state.missing_slots:
            return calc, calc_hint

        calc_started = time.perf_counter()
        calc = await self._compute_with_calculator_from_ledger(state)
        if not calc:
            return calc, calc_hint

        if calc.get("ok"):
            calc_hint = (
                f"expression={calc.get('expression')}, "
                f"result={calc.get('result')}"
            )
        else:
            calc_hint = f"calculator_error={calc.get('error', '')}"
        append_trace(
            state.trace,
            step="execution_compute_tool",
            input={
                "query_state": state.query_state,
                "ledger_entries": len(state.evidence_ledger),
            },
            output=calc,
            duration_ms=(time.perf_counter() - calc_started) * 1000.0,
        )
        return calc, calc_hint

    def _apply_compute_direct_answer_before_synthesis(
        self,
        state: AgentState,
        answer_type: str,
        calc: Optional[dict[str, Any]],
        forced_started: float,
    ) -> bool:
        numeric_compute = self._is_numeric_compute_query(state.user_query, state.query_state)
        if not (answer_type == "compute" and numeric_compute and calc and calc.get("ok")):
            return False

        calc_result = str(calc.get("result", "") or "").strip()
        if not calc_result:
            return False

        candidate = self._build_calc_result_answer(
            query_state=state.query_state,
            evidence_ledger=state.evidence_ledger,
            context=state.context,
            calc_result=calc_result,
        )
        grounded, reason = self._verify_answer_grounding(
            answer=candidate,
            query_state=state.query_state,
            evidence_ledger=state.evidence_ledger,
            context=state.context,
            missing_slots=state.missing_slots,
        )
        if grounded:
            state.final_answer = candidate
            append_trace(
                state.trace,
                step="execution_compute_direct_answer",
                input={
                    "calculator_result": calc_result,
                    "query_state": state.query_state,
                },
                output=state.final_answer,
            )
            append_trace(
                state.trace,
                step="execution_forced_synthesis",
                input={"mode": "calculator_direct"},
                output={
                    "final_answer": state.final_answer,
                    "attempts": [
                        {
                            "attempt": 1,
                            "accepted": True,
                            "reason": "calculator_direct",
                        }
                    ],
                    "compute_override_applied": True,
                    "compute_override_reason": "calculator_direct",
                },
                duration_ms=(time.perf_counter() - forced_started) * 1000.0,
            )
            return True

        append_trace(
            state.trace,
            step="execution_compute_direct_answer_skipped",
            input={
                "calculator_result": calc_result,
                "candidate_answer": candidate,
            },
            output={"reason": reason},
        )
        return False

    def _apply_execution_override(
        self,
        *,
        state: AgentState,
        override_answer: Optional[str],
        trace_step: str,
    ) -> bool:
        if not override_answer or override_answer == state.final_answer:
            return False
        previous = state.final_answer
        state.final_answer = override_answer
        append_trace(
            state.trace,
            step=trace_step,
            input={"before_answer": previous},
            output={"after_answer": state.final_answer},
        )
        return True

    def _maybe_apply_compute_result_override(
        self,
        *,
        state: AgentState,
        calc: Optional[dict[str, Any]],
    ) -> tuple[bool, str]:
        if not (
            str(state.query_state.get("answer_type", "")).lower() == "compute"
            and not state.missing_slots
            and calc
            and calc.get("ok")
        ):
            return False, ""

        calc_result = str(calc.get("result", "") or "").strip()
        if not calc_result or self._answer_matches_calc_result(state.final_answer, calc_result):
            return False, ""

        candidate = self._build_calc_result_answer(
            query_state=state.query_state,
            evidence_ledger=state.evidence_ledger,
            context=state.context,
            calc_result=calc_result,
        )
        grounded, reason = self._verify_answer_grounding(
            answer=candidate,
            query_state=state.query_state,
            evidence_ledger=state.evidence_ledger,
            context=state.context,
            missing_slots=state.missing_slots,
        )
        if grounded:
            previous = state.final_answer
            state.final_answer = candidate
            append_trace(
                state.trace,
                step="execution_compute_result_override",
                input={
                    "previous_answer": previous,
                    "calculator_result": calc_result,
                },
                output=state.final_answer,
            )
            return True, "calculator_result_mismatch"

        append_trace(
            state.trace,
            step="execution_compute_result_override_skipped",
            input={
                "current_answer": state.final_answer,
                "calculator_result": calc_result,
                "candidate_answer": candidate,
            },
            output={
                "reason": reason,
            },
        )
        return False, f"override_candidate_rejected:{reason}"

    def _apply_post_synthesis_overrides(self, state: AgentState) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for key_prefix, builder_name, trace_step, applied_reason in self._POST_SYNTHESIS_OVERRIDES:
            builder = getattr(self, builder_name, None)
            override_answer = builder(state) if builder else None
            applied = self._apply_execution_override(
                state=state,
                override_answer=override_answer,
                trace_step=trace_step,
            )
            meta[f"{key_prefix}_applied"] = applied
            meta[f"{key_prefix}_reason"] = applied_reason if applied else ""
        return meta

    async def _run_forced_synthesis_if_needed(self, state: AgentState) -> None:
        if state.final_answer:
            return
        forced_started = time.perf_counter()
        answer_type = str(state.query_state.get("answer_type", "")).lower()
        if answer_type == "compute" and state.missing_slots:
            self._run_compute_slot_fill_before_synthesis(state)

        if answer_type == "compute" and not state.missing_slots:
            self._run_compute_slot_realign_before_synthesis(state)

        calc, calc_hint = await self._run_compute_calculator_before_synthesis(
            state=state,
            answer_type=answer_type,
        )
        if self._apply_compute_direct_answer_before_synthesis(
            state=state,
            answer_type=answer_type,
            calc=calc,
            forced_started=forced_started,
        ):
            return
        synthesis_prompt = self._synthesis_prompt_template().format(
            query_state=self._compact_json(state.query_state, max_chars=1200),
            evidence_ledger=self._compact_json({
                "entries": state.evidence_ledger,
                "missing_slots": state.missing_slots,
            }, max_chars=2000),
            context=state.context if state.context.strip() else "No retrieved context.",
        )
        if calc_hint:
            synthesis_prompt += f"\n\nCALCULATOR_RESULT:\n{calc_hint}"
        synthesis_prompt += f"\n\nQUERY:\n{state.user_query}"
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_FORMAT_INSTRUCTION},
            {"role": "user", "content": synthesis_prompt},
            {"role": "user", "content": FINAL_ANSWER_FORMAT_INSTRUCTION},
        ]
        started = time.perf_counter()
        state.final_answer, synthesis_attempts = await self._generate_single_final_answer(
            messages,
            stage="synthesis",
            state=state,
            max_attempts=3,
        )

        override_applied, override_reason = self._maybe_apply_compute_result_override(
            state=state,
            calc=calc,
        )

        normalized_answer = self._normalize_final_answer_for_query(
            state.final_answer,
            state.query_state,
        )
        if normalized_answer != state.final_answer:
            append_trace(
                state.trace,
                step="execution_answer_normalization",
                input={"before": state.final_answer},
                output={"after": normalized_answer, "reason": "query_metric_normalization"},
            )
            state.final_answer = normalized_answer

        override_meta = self._apply_post_synthesis_overrides(state)

        append_trace(
            state.trace,
            step="execution_forced_synthesis",
            input=messages,
            output={
                "final_answer": state.final_answer,
                "attempts": synthesis_attempts,
                "compute_override_applied": override_applied,
                "compute_override_reason": override_reason,
                **override_meta,
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
