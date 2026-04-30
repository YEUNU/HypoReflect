import logging
import time
from typing import Any

from utils.prompts import (
    PERCEPTION_PROMPT,
    PERCEPTION_FORMAT_INSTRUCTION,
    PERCEPTION_RETRY_PROMPT,
)
from models.hyporeflect.state import AgentState
from core.config import RAGConfig
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class PerceptionHandler:
    _MAX_JSON_ATTEMPTS = 3

    def __init__(self, llm, stage_model: str = ""):
        self.llm = llm
        self.stage_model = stage_model or RAGConfig.PERCEPTION_MODEL

    @staticmethod
    def _validate_perception_json(data: dict[str, Any]) -> tuple[bool, str]:
        complexity = str(data.get("complexity", "") or "").strip().lower()
        if complexity not in {"simple", "complex"}:
            return False, "complexity must be simple|complex"
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            return False, "reason must be string"
        return True, ""

    @staticmethod
    def _retry_message(failed_output: Any, reason: str) -> str:
        return PERCEPTION_RETRY_PROMPT.format(
            error=reason,
            previous_output=compact_json(failed_output, max_chars=900),
        )

    async def run(self, state: AgentState):
        started = time.perf_counter()
        base_messages = [
            {"role": "user", "content": PERCEPTION_PROMPT.format(query=state.user_query)},
            {"role": "user", "content": PERCEPTION_FORMAT_INSTRUCTION},
        ]
        data, ok, attempts = await generate_json_with_retries(
            self.llm,
            base_messages,
            self._validate_perception_json,
            self._retry_message,
            max_attempts=self._MAX_JSON_ATTEMPTS,
            logger=logger,
            warning_prefix="perception json generation failed",
            model=self.stage_model,
        )

        state.intent = "research"
        if ok:
            complexity = str(data.get("complexity", "complex")).strip().lower()
            state.is_complex = complexity == "complex"
            parsed_output = {
                "complexity": "complex" if state.is_complex else "simple",
                "reason": str(data.get("reason", "") or ""),
                "attempts": attempts,
            }
        else:
            state.is_complex = True
            parsed_output = {
                "complexity": "complex",
                "reason": "schema_validation_failed",
                "attempts": attempts,
            }
        append_trace(
            state.trace,
            step="perception",
            input=base_messages,
            output=parsed_output,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        return None
