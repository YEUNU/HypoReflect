import logging
import time
from typing import Any

from utils.prompts import (
    PLANNING_PROMPT,
    PLANNING_FILTER_PROMPT,
    PLANNING_FILTER_FORMAT_INSTRUCTION,
    PLANNING_FILTER_RETRY_PROMPT,
)
from models.hyporeflect.state import AgentState
from core.config import RAGConfig
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class PlanningHandler:
    _MAX_JSON_ATTEMPTS = 3

    def __init__(self, llm, stage_model: str = ""):
        self.llm = llm
        self.stage_model = stage_model or RAGConfig.PLANNING_MODEL

    @staticmethod
    def _planning_prompt() -> str:
        return PLANNING_PROMPT

    @staticmethod
    def _planning_filter_prompt() -> str:
        return PLANNING_FILTER_PROMPT

    @staticmethod
    def _default_filter_policy() -> dict[str, Any]:
        return {
            "must_match": {
                "entity": True,
                "period": True,
                "source_anchor": "soft",
            },
            "preferred_markers": [],
            "disallowed_patterns": [],
            "slot_conflict_strategy": "best_supported",
        }

    @staticmethod
    def _normalize_filter_policy(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return PlanningHandler._default_filter_policy()
        must_match = data.get("must_match", {})
        if not isinstance(must_match, dict):
            must_match = PlanningHandler._default_filter_policy()["must_match"]
        default_source_anchor = "soft"
        source_anchor = str(must_match.get("source_anchor", default_source_anchor)).strip().lower()
        if source_anchor not in {"strict", "soft", "none"}:
            source_anchor = default_source_anchor
        return {
            "must_match": {
                "entity": bool(must_match.get("entity", True)),
                "period": bool(must_match.get("period", True)),
                "source_anchor": source_anchor,
            },
            "preferred_markers": [
                str(x).strip() for x in data.get("preferred_markers", []) if str(x).strip()
            ] if isinstance(data.get("preferred_markers", []), list) else [],
            "disallowed_patterns": [
                str(x).strip() for x in data.get("disallowed_patterns", []) if str(x).strip()
            ] if isinstance(data.get("disallowed_patterns", []), list) else [],
            "slot_conflict_strategy": str(
                data.get("slot_conflict_strategy", "best_supported")
            ).strip().lower() or "best_supported",
        }

    @staticmethod
    def _validate_filter_policy_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        must_match = data.get("must_match")
        if not isinstance(must_match, dict):
            return False, "must_match must be object"
        if not isinstance(must_match.get("entity"), bool):
            return False, "must_match.entity must be boolean"
        if not isinstance(must_match.get("period"), bool):
            return False, "must_match.period must be boolean"
        source_anchor = str(must_match.get("source_anchor", "") or "").strip().lower()
        if source_anchor not in {"strict", "soft", "none"}:
            return False, "must_match.source_anchor must be strict|soft|none"

        preferred_markers = data.get("preferred_markers")
        if not isinstance(preferred_markers, list) or any(not isinstance(x, str) for x in preferred_markers):
            return False, "preferred_markers must be string array"

        disallowed_patterns = data.get("disallowed_patterns")
        if not isinstance(disallowed_patterns, list) or any(not isinstance(x, str) for x in disallowed_patterns):
            return False, "disallowed_patterns must be string array"

        strategy = str(data.get("slot_conflict_strategy", "") or "").strip().lower()
        if strategy not in {"best_supported", "keep_missing_on_tie"}:
            return False, "slot_conflict_strategy must be best_supported|keep_missing_on_tie"
        return True, ""

    @staticmethod
    def _retry_message(failed_output: Any, reason: str) -> str:
        return PLANNING_FILTER_RETRY_PROMPT.format(
            error=reason,
            previous_output=compact_json(failed_output, max_chars=900),
        )

    async def run(self, state: AgentState):
        plan_started = time.perf_counter()
        messages = [{"role": "user", "content": self._planning_prompt().format(query=state.user_query, context="")}]
        _plan_kwargs = dict(apply_default_sampling=False)
        if self.stage_model:
            _plan_kwargs["model"] = self.stage_model
        state.plan = await self.llm.generate_response(messages, **_plan_kwargs)
        append_trace(
            state.trace,
            step="planning",
            input=messages,
            output=state.plan,
            duration_ms=(time.perf_counter() - plan_started) * 1000.0,
        )
        policy_messages = [
            {
                "role": "user",
                "content": self._planning_filter_prompt().format(query=state.user_query, plan=state.plan),
            },
            {"role": "user", "content": PLANNING_FILTER_FORMAT_INSTRUCTION},
        ]
        filter_started = time.perf_counter()
        policy_raw, ok, attempts = await generate_json_with_retries(
            self.llm,
            policy_messages,
            self._validate_filter_policy_json,
            self._retry_message,
            max_attempts=self._MAX_JSON_ATTEMPTS,
            logger=logger,
            warning_prefix="planning filter json generation failed",
            model=self.stage_model,
        )
        state.filter_policy = self._normalize_filter_policy(policy_raw)
        append_trace(
            state.trace,
            step="planning_filter",
            input=policy_messages,
            output={
                "filter_policy": state.filter_policy,
                "accepted": ok,
                "attempts": attempts,
            },
            duration_ms=(time.perf_counter() - filter_started) * 1000.0,
        )
