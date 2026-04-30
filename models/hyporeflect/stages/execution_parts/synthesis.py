import logging
from typing import Any

from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import extract_final_answer_from_json
from models.hyporeflect.stages.llm_json import generate_json_with_retries
from models.hyporeflect.trace import append_trace
from utils.prompts import FINAL_ANSWER_RETRY_PROMPT


logger = logging.getLogger(__name__)


class SynthesisSupport:
    def _final_answer_retry_message(
        self,
        stage: str,
        failed_output: Any,
        reason: str,
    ) -> str:
        prev = self._compact_json(failed_output, max_chars=900)
        return FINAL_ANSWER_RETRY_PROMPT.format(
            stage=stage,
            reason=reason,
            previous_output=prev,
        )

    async def _generate_single_final_answer(
        self,
        base_messages: list[dict[str, str]],
        stage: str,
        state: AgentState,
        max_attempts: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        def validate(data: dict[str, Any]) -> tuple[bool, str]:
            answer, reason = extract_final_answer_from_json(data)
            if not answer:
                return False, reason
            return self._verify_answer_grounding(
                answer=answer,
                query_state=state.query_state,
                evidence_ledger=state.evidence_ledger,
                context=state.context,
                missing_slots=state.missing_slots,
            )

        def retry_message(data: dict[str, Any], reason: str) -> str:
            return self._final_answer_retry_message(
                stage=stage,
                failed_output=data,
                reason=reason,
            )

        data, ok, attempts = await generate_json_with_retries(
            self.llm,
            base_messages,
            validate,
            retry_message,
            max_attempts=max_attempts,
            logger=logger,
            warning_prefix=f"{stage} json generation failed",
            model=self.stage_model,
        )
        if ok:
            answer, _ = extract_final_answer_from_json(data)
            if answer:
                return answer, attempts
        return "@@ANSWER: insufficient evidence", attempts

    def _should_terminate_expansion(self, state: AgentState, loop_state: Any) -> bool:
        coverage_complete = False
        required_slots = self._required_slots(state.query_state)
        if required_slots:
            if not state.missing_slots and len(state.evidence_ledger) >= len(required_slots):
                coverage_complete = True
        elif state.evidence_ledger:
            coverage_complete = True
        if coverage_complete and loop_state.tool_calls_used > 1:
            return True
        if loop_state.tool_calls_used >= loop_state.max_tool_calls:
            return True
        return False

    def _build_expansion_messages(self, state: AgentState) -> list[dict[str, str]]:
        context_tail = (
            self._extract_relevant_span(
                state.context,
                query_state=state.query_state,
                max_chars=min(self.context_char_budget, 3200),
            )
            if state.context
            else ""
        )
        return [
            {
                "role": "system",
                "content": self._agent_execution_prompt_template().format(
                    query_state=self._compact_json(state.query_state, max_chars=900),
                    missing_slots=self._compact_json(state.missing_slots, max_chars=600),
                    context=context_tail if context_tail.strip() else "No retrieved context yet.",
                ),
            },
            {"role": "user", "content": f"Query: {state.user_query}\nPlan: {state.plan}"},
        ]

    def _handle_direct_response(self, state: AgentState, turn: int, resp: Any) -> bool:
        candidate_answer = str(resp)
        if (
            self._is_numeric_compute_query(state.user_query, state.query_state)
            and state.missing_slots
        ):
            append_trace(
                state.trace,
                step=f"compute_answer_blocked_{turn}",
                input=candidate_answer,
                output={
                    "reason": "missing_slots_nonempty",
                    "missing_slots": state.missing_slots,
                },
            )
            return False
        if self._is_insufficient_answer(candidate_answer) and not state.missing_slots:
            append_trace(
                state.trace,
                step=f"insufficient_blocked_{turn}",
                input=candidate_answer,
                output={
                    "reason": "required_slots_already_grounded",
                    "missing_slots": state.missing_slots,
                },
            )
            return True
        grounded, reason = self._verify_answer_grounding(
            answer=candidate_answer,
            query_state=state.query_state,
            evidence_ledger=state.evidence_ledger,
            context=state.context,
            missing_slots=state.missing_slots,
        )
        if not grounded:
            append_trace(
                state.trace,
                step=f"direct_answer_rejected_{turn}",
                input=candidate_answer,
                output={
                    "reason": reason,
                    "missing_slots": state.missing_slots,
                },
            )
            return True
        state.final_answer = candidate_answer
        return True
