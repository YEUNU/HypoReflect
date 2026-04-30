import logging
import time
from typing import Any

from utils.prompts import (
    RESPONSE_REFINEMENT_PROMPT,
    FINAL_ANSWER_FORMAT_INSTRUCTION,
    REFINEMENT_RETRY_PROMPT,
)
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import extract_final_answer_from_json
from core.config import RAGConfig
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class RefinementHandler:
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
