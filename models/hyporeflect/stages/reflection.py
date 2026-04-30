import logging
import re
import time
from typing import Any

from core.config import RAGConfig
from utils.prompts import (
    REFLECTION_PROMPT,
    REFLECTION_FORMAT_INSTRUCTION,
    REFLECTION_RETRY_PROMPT,
)
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import (
    _ANSWER_PREFIX_RE,
    CITATION_RE,
    NUMERIC_METRIC_KEYS,
    NUMERIC_QUERY_MARKERS,
    extract_first_number,
    missing_data_policy,
)
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class ReflectionHandler:
    _AUDIT_ZERO_POLICY = (
        "\nSTRICT_AUDIT: Query policy requires @@ANSWER: 0 when value is not explicitly outlined."
    )
    _AUDIT_INAPPLICABLE_POLICY = (
        "\nSTRICT_AUDIT: Query policy requires inapplicable-metric explanation, not insufficient evidence."
    )
    _AUDIT_LEDGER_DIAGNOSTIC = (
        "\nSTRICT_AUDIT: insufficient evidence requires ledger attempt diagnostics; "
        "no valid evidence-attempt trace was found."
    )
    _AUDIT_MISSING_CITATION = (
        "\nSTRICT_AUDIT: Missing mandatory inline citations "
        "([[Title, Page X, Chunk Y]] or legacy [[Title, sent_id]])."
    )
    _AUDIT_MULTIPLE_ANSWER_PREFIX = (
        "\nSTRICT_AUDIT: Multiple @@ANSWER prefixes detected in final answer."
    )
    _AUDIT_ARITHMETIC_MISMATCH = (
        "\nSTRICT_AUDIT: Deterministic arithmetic check failed ({reason})."
    )

    def __init__(self, llm, stage_model: str = ""):
        self.llm = llm
        self.stage_model = stage_model or RAGConfig.REFLECTION_MODEL

    @staticmethod
    def _reflection_prompt() -> str:
        return REFLECTION_PROMPT

    @staticmethod
    def _is_zero_policy_answer(answer: str) -> bool:
        text = str(answer or "").strip()
        text = CITATION_RE.sub("", text)
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return normalized in {"@@answer: 0", "@@answer: 0.0", "@@answer: 0.00"}

    def _allow_no_citation(self, state: AgentState) -> bool:
        return (
            missing_data_policy(state) == "zero_if_not_explicit"
            and bool(state.missing_slots)
            and self._is_zero_policy_answer(state.final_answer)
        )

    @staticmethod
    def _validate_critique_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        decision = str(data.get("decision", "") or "").strip().upper()
        if decision not in {"PASS", "FAIL"}:
            return False, "decision must be PASS|FAIL"
        arithmetic = str(data.get("arithmetic_check", "") or "").strip().lower()
        if arithmetic not in {"ok", "fail", "na"}:
            return False, "arithmetic_check must be ok|fail|na"
        issues_raw = data.get("issues", [])
        if not isinstance(issues_raw, list) or any(not isinstance(item, str) for item in issues_raw):
            return False, "issues must be string array"
        issues = [str(item).strip() for item in issues_raw if str(item).strip()]
        if decision == "FAIL" and not issues and arithmetic != "fail":
            return False, "FAIL requires non-empty issues or arithmetic_check=fail"
        return True, ""

    @staticmethod
    def _retry_message(failed_output: Any, reason: str) -> str:
        return REFLECTION_RETRY_PROMPT.format(
            error=reason,
            previous_output=compact_json(failed_output, max_chars=900),
        )

    @staticmethod
    def _normalize_critique_json(data: dict[str, Any]) -> str:
        decision = str(data.get("decision", "") or "").strip().upper()
        if decision == "PASS":
            return "PASS"
        issues_raw = data.get("issues", [])
        issues: list[str] = []
        if isinstance(issues_raw, list):
            for item in issues_raw:
                text = str(item).strip()
                if not text:
                    continue
                issues.append(f"- {text}" if not text.startswith("-") else text)
                if len(issues) >= 3:
                    break
        arithmetic = str(data.get("arithmetic_check", "na") or "na").strip().lower()
        if arithmetic == "fail" and len(issues) < 3:
            issues.append("- arithmetic check failed")
        if not issues:
            issues = ["- issue 1: invalid reflection json"]
        return "FAIL\n" + "\n".join(issues)

    @staticmethod
    def _build_reflection_meta(result: dict[str, Any], accepted: bool) -> dict[str, Any]:
        issues_raw = result.get("issues", []) if isinstance(result.get("issues", []), list) else []
        return {
            "decision": str(result.get("decision", "") or "").strip().upper(),
            "arithmetic_check": str(result.get("arithmetic_check", "na") or "na").strip().lower(),
            "issues": [str(item).strip() for item in issues_raw if str(item).strip()],
            "accepted": bool(accepted),
        }

    @staticmethod
    def _has_diagnostic_signal(diagnostics: dict[str, Any]) -> bool:
        entries_raw = int(diagnostics.get("entries_raw", 0) or 0)
        accepted_entries = int(diagnostics.get("accepted_entries", 0) or 0)
        reject_reasons = diagnostics.get("reject_reasons", {})
        if entries_raw > 0 or accepted_entries > 0:
            return True
        return isinstance(reject_reasons, dict) and bool(reject_reasons)

    def _has_valid_ledger_attempt_trace(self, state: AgentState) -> bool:
        attempts = state.ledger_attempts if isinstance(state.ledger_attempts, list) else []
        has_attempt_log = False
        has_diagnostic_signal = False
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            has_attempt_log = True
            diagnostics = attempt.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                continue
            if self._has_diagnostic_signal(diagnostics):
                has_diagnostic_signal = True
                break
        return has_attempt_log and has_diagnostic_signal

    @staticmethod
    def _normalize_metric_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"(?<!\d)(?:fy\s*)?(?:19|20)\d{2}(?!\d)", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _is_numeric_compute_query(self, state: AgentState) -> bool:
        answer_type = str((state.query_state or {}).get("answer_type", "") or "").strip().lower()
        if answer_type != "compute":
            return False
        query_lower = str(state.user_query or "").strip().lower()
        metric_key = self._normalize_metric_key((state.query_state or {}).get("metric", ""))
        if any(marker in query_lower for marker in NUMERIC_QUERY_MARKERS):
            return True
        return metric_key in NUMERIC_METRIC_KEYS

    def _answer_matches_calc_result(self, answer: str, calc_result: str) -> bool:
        answer_text = str(answer or "").strip().replace(",", "")
        result_text = str(calc_result or "").strip().replace(",", "")
        if not answer_text or not result_text:
            return False
        if result_text in answer_text:
            return True
        answer_num = extract_first_number(answer_text)
        result_num = extract_first_number(result_text)
        if answer_num is None or result_num is None:
            return False
        decimals = 0
        if "." in result_text:
            decimals = len(result_text.rsplit(".", 1)[1])
        tolerance = 10 ** (-(decimals + 1)) if decimals > 0 else 1e-6
        return abs(answer_num - result_num) <= tolerance

    @staticmethod
    def _latest_calculator_result(state: AgentState) -> str:
        trace = state.trace if isinstance(state.trace, list) else []
        for event in reversed(trace):
            if not isinstance(event, dict):
                continue
            step = str(event.get("step", "") or "")
            output = event.get("output", {})
            if not isinstance(output, dict):
                continue
            if step == "execution_compute_tool" and output.get("ok"):
                return str(output.get("result", "") or "").strip()
            if step.startswith("calculator_update_") and output.get("ok"):
                return str(output.get("result", "") or "").strip()
        return ""

    @staticmethod
    def _decision_is_pass(critique: str) -> bool:
        decision_line = critique.splitlines()[0].strip().upper() if critique else ""
        return decision_line == "PASS"

    @staticmethod
    def _has_multiple_answer_prefix(answer: str) -> bool:
        return len(_ANSWER_PREFIX_RE.findall(answer or "")) > 1

    def _build_reflection_messages(self, state: AgentState) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": self._reflection_prompt().format(
                    query=state.user_query,
                    query_state=compact_json(state.query_state, max_chars=1200),
                    evidence_ledger=compact_json(
                        {
                            "entries": state.evidence_ledger,
                            "missing_slots": state.missing_slots,
                        },
                        max_chars=2000,
                    ),
                    context=state.context,
                    draft_answer=state.final_answer,
                ),
            },
            {"role": "user", "content": REFLECTION_FORMAT_INSTRUCTION},
        ]

    async def _generate_reflection_result(
        self,
        base_messages: list[dict[str, str]],
    ) -> tuple[Any, bool, int]:
        data, ok, attempts = await generate_json_with_retries(
            self.llm,
            base_messages,
            self._validate_critique_json,
            self._retry_message,
            max_attempts=3,
            logger=logger,
            warning_prefix="reflection json generation failed",
            model=self.stage_model,
        )
        if not ok:
            data = {"decision": "FAIL", "issues": ["invalid reflection json"], "arithmetic_check": "na"}
        return data, ok, attempts

    def _initial_reflection_gate(
        self,
        state: AgentState,
    ) -> tuple[bool, bool, bool, bool]:
        has_citations = CITATION_RE.search(state.final_answer) is not None
        has_multiple_final_answers = self._has_multiple_answer_prefix(state.final_answer)
        allow_no_citation = self._allow_no_citation(state)

        passed = self._decision_is_pass(state.critique)
        if RAGConfig.STRICT_CITATION_CHECK:
            passed = passed and (has_citations or allow_no_citation)
        if has_multiple_final_answers:
            passed = False
        return passed, has_citations, allow_no_citation, has_multiple_final_answers

    def _apply_insufficient_policy_audit(
        self,
        state: AgentState,
        passed: bool,
        answer_insufficient: bool,
    ) -> bool:
        if not (answer_insufficient and state.missing_slots):
            return passed

        policy = missing_data_policy(state)
        if policy == "zero_if_not_explicit":
            passed = False
            state.critique += self._AUDIT_ZERO_POLICY
        elif policy == "inapplicable_explain":
            passed = False
            state.critique += self._AUDIT_INAPPLICABLE_POLICY

        if not self._has_valid_ledger_attempt_trace(state):
            passed = False
            state.critique += self._AUDIT_LEDGER_DIAGNOSTIC
        return passed

    def _apply_arithmetic_audit(
        self,
        state: AgentState,
        passed: bool,
        answer_insufficient: bool,
    ) -> bool:
        if not self._is_numeric_compute_query(state):
            return passed

        calc_result = self._latest_calculator_result(state)
        if not calc_result:
            return passed

        arithmetic_reason = ""
        if answer_insufficient and not state.missing_slots:
            arithmetic_reason = "numeric_compute_returned_insufficient_despite_grounded_slots"
        elif not self._answer_matches_calc_result(state.final_answer, calc_result):
            arithmetic_reason = "numeric_compute_answer_mismatch_with_calculator_result"

        state.reflection_meta["deterministic_arithmetic_applied"] = True
        state.reflection_meta["deterministic_calculator_result"] = calc_result
        if arithmetic_reason:
            passed = False
            state.reflection_meta["arithmetic_check"] = "fail"
            issues = state.reflection_meta.get("issues", [])
            if isinstance(issues, list):
                issues.append(arithmetic_reason)
                state.reflection_meta["issues"] = issues
            state.critique += self._AUDIT_ARITHMETIC_MISMATCH.format(reason=arithmetic_reason)
        elif str(state.reflection_meta.get("arithmetic_check", "na")).strip().lower() != "fail":
            state.reflection_meta["arithmetic_check"] = "ok"
        return passed

    async def run(self, state: AgentState) -> bool:
        started = time.perf_counter()
        base_messages = self._build_reflection_messages(state)
        data, ok, attempts = await self._generate_reflection_result(base_messages)
        raw_obj: dict[str, Any] = data if isinstance(data, dict) else {}
        state.reflection_meta = self._build_reflection_meta(raw_obj, ok)
        state.critique = self._normalize_critique_json(data)
        append_trace(
            state.trace,
            step="reflection",
            input=base_messages,
            output={
                "normalized": state.critique,
                "raw": data,
                "meta": state.reflection_meta,
                "accepted": ok,
                "attempts": attempts,
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        passed, has_citations, allow_no_citation, has_multiple_final_answers = self._initial_reflection_gate(state)

        answer_insufficient = "insufficient evidence" in str(state.final_answer or "").lower()
        passed = self._apply_insufficient_policy_audit(
            state=state,
            passed=passed,
            answer_insufficient=answer_insufficient,
        )
        passed = self._apply_arithmetic_audit(
            state=state,
            passed=passed,
            answer_insufficient=answer_insufficient,
        )

        if not passed and not has_citations and not allow_no_citation:
            state.critique += self._AUDIT_MISSING_CITATION
        if has_multiple_final_answers:
            state.critique += self._AUDIT_MULTIPLE_ANSWER_PREFIX
        return passed
