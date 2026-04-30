import logging
import re
import time
from typing import Any, Optional

from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import normalize_missing_data_policy
from models.hyporeflect.stages.llm_json import generate_json_with_retries
from models.hyporeflect.trace import append_trace
from utils.prompts import QUERY_STATE_FORMAT_INSTRUCTION, QUERY_STATE_RETRY_PROMPT


logger = logging.getLogger(__name__)


class QueryStateSupport:
    def _apply_query_state_heuristics(
        self,
        query: str,
        query_state: dict[str, Any],
    ) -> dict[str, Any]:
        query_lower = str(query or "").strip().lower()
        if not query_lower:
            return query_state
        adjusted = dict(query_state or {})
        answer_type = str(adjusted.get("answer_type", "") or "").strip().lower()
        if answer_type == "extract":
            if re.match(r"^(is|are|was|were|does|do|did|has|have|can|could|should|would)\b", query_lower):
                adjusted["answer_type"] = "boolean"
            elif re.match(r"^(which|what are|list|name)\b", query_lower):
                adjusted["answer_type"] = "list"

        answer_type = str(adjusted.get("answer_type", "") or "").strip().lower()
        metric_text = str(adjusted.get("metric", "") or "").strip()
        metric_key = self._canonical_metric_key(metric_text)
        has_explicit_anchor = self._query_has_explicit_statement_anchor(query_lower)
        explicit_anchors = self._extract_query_statement_anchors(query_lower)

        if has_explicit_anchor and explicit_anchors:
            current_anchor = self._normalize_source_anchor(adjusted.get("source_anchor"))
            if not current_anchor:
                if len(explicit_anchors) == 1:
                    current_anchor = explicit_anchors[0]
                else:
                    inferred_top = self._infer_anchor_for_metric(metric_text)
                    if inferred_top in explicit_anchors:
                        current_anchor = inferred_top
            if current_anchor:
                adjusted["source_anchor"] = current_anchor

            slots_raw = adjusted.get("required_slots", [])
            if isinstance(slots_raw, list) and slots_raw:
                backfilled_slots: list[Any] = []
                for slot in slots_raw:
                    struct = self._parse_slot_struct(slot)
                    if not struct:
                        backfilled_slots.append(slot)
                        continue
                    slot_anchor = self._normalize_source_anchor(struct.get("source_anchor"))
                    if slot_anchor:
                        backfilled_slots.append(struct)
                        continue
                    inferred_slot_anchor = self._infer_anchor_for_metric(struct.get("metric", ""))
                    if inferred_slot_anchor and inferred_slot_anchor in explicit_anchors:
                        struct["source_anchor"] = inferred_slot_anchor
                    elif current_anchor:
                        struct["source_anchor"] = current_anchor
                    backfilled_slots.append(struct)
                adjusted["required_slots"] = backfilled_slots

        if (
            not has_explicit_anchor
            and answer_type in {"extract", "boolean", "list"}
            and "dividend" in metric_key
        ):
            if self._normalize_source_anchor(adjusted.get("source_anchor")) == "cash flow statement":
                adjusted["source_anchor"] = None
            slots = adjusted.get("required_slots", [])
            if isinstance(slots, list):
                relaxed_slots: list[Any] = []
                for slot in slots:
                    struct = self._parse_slot_struct(slot)
                    if not struct:
                        relaxed_slots.append(slot)
                        continue
                    if self._is_dividend_metric(struct.get("metric", "")) and str(
                        struct.get("source_anchor", "") or ""
                    ).strip().lower() == "cash flow statement":
                        struct.pop("source_anchor", None)
                    relaxed_slots.append(struct)
                adjusted["required_slots"] = relaxed_slots

        if "quick ratio" in metric_key or "quick ratio" in query_lower:
            adjusted["metric"] = "quick ratio"
            if re.match(r"^(is|are|was|were|does|do|did|has|have|can|could|should|would)\b", query_lower):
                adjusted["answer_type"] = "boolean"
            base_entity = str(adjusted.get("entity", "") or "").strip().lower()
            base_period = str(adjusted.get("period", "") or "").strip().lower()
            quick_slot: dict[str, Any] = {"metric": "quick ratio"}
            if base_entity:
                quick_slot["entity"] = base_entity
            if base_period:
                quick_slot["period"] = base_period
            anchor = self._normalize_source_anchor(adjusted.get("source_anchor"))
            if has_explicit_anchor and anchor:
                quick_slot["source_anchor"] = anchor
                adjusted["source_anchor"] = anchor
            else:
                adjusted["source_anchor"] = None
            adjusted["required_slots"] = [quick_slot]

        if (
            ("capital intensity" in metric_key or "capital-intensive" in query_lower or "capital intensive" in query_lower)
            and re.match(r"^(is|are|was|were|does|do|did|has|have|can|could|should|would)\b", query_lower)
        ):
            adjusted["answer_type"] = "boolean"
            adjusted["metric"] = "capital intensity"
            base_entity = str(adjusted.get("entity", "") or "").strip().lower()
            base_period = str(adjusted.get("period", "") or "").strip().lower()
            capex_slot: dict[str, Any] = {"metric": "capital expenditures"}
            revenue_slot: dict[str, Any] = {"metric": "revenue"}
            if base_entity:
                capex_slot["entity"] = base_entity
                revenue_slot["entity"] = base_entity
            if base_period:
                capex_slot["period"] = base_period
                revenue_slot["period"] = base_period
            capex_slot["source_anchor"] = "cash flow statement"
            revenue_slot["source_anchor"] = "income statement"
            adjusted["source_anchor"] = None
            adjusted["required_slots"] = [capex_slot, revenue_slot]

        driver_intent = any(
            marker in query_lower
            for marker in ["what drove", "what drives", "driven by", "driver", "drivers", "caused by"]
        )
        if driver_intent and answer_type in {"extract", "list"} and metric_text:
            slot_source_anchor = self._normalize_source_anchor(adjusted.get("source_anchor"))
            driver_metric = f"{metric_text} change drivers"
            slots_raw = adjusted.get("required_slots", [])
            slots = list(slots_raw) if isinstance(slots_raw, list) else []
            has_driver_slot = False
            for slot in slots:
                struct = self._parse_slot_struct(slot)
                if (
                    struct
                    and "driver" in self._canonical_metric_key(struct.get("metric", ""))
                    and self._metric_matches(struct.get("metric", ""), driver_metric)
                ):
                    has_driver_slot = True
                    break
            if not has_driver_slot:
                driver_slot: dict[str, Any] = {
                    "entity": str(adjusted.get("entity", "") or "").strip().lower(),
                    "period": str(adjusted.get("period", "") or "").strip().lower(),
                    "metric": driver_metric.lower(),
                }
                if slot_source_anchor:
                    driver_slot["source_anchor"] = slot_source_anchor
                slots.append(driver_slot)
                adjusted["required_slots"] = slots

        if (
            "exclude" in query_lower
            and any(tok in query_lower for tok in ["m&a", "acquisition", "divestiture"])
            and "segment" in query_lower
            and any(tok in metric_key for tok in ["segment growth", "growth impact"])
        ):
            adjusted_metric = "organic sales change by business segment excluding acquisitions/divestitures"
            adjusted["metric"] = adjusted_metric
            slots_raw = adjusted.get("required_slots", [])
            if isinstance(slots_raw, list):
                rewritten_slots: list[Any] = []
                for slot in slots_raw:
                    struct = self._parse_slot_struct(slot)
                    if not struct:
                        rewritten_slots.append(slot)
                        continue
                    if any(tok in self._canonical_metric_key(struct.get("metric", "")) for tok in ["segment growth", "growth impact"]):
                        struct["metric"] = adjusted_metric.lower()
                    rewritten_slots.append(struct)
                adjusted["required_slots"] = rewritten_slots
        return adjusted

    def _sanitize_query_state(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        answer_type = str(data.get("answer_type", "extract")).strip().lower()
        if answer_type not in self._VALID_ANSWER_TYPES:
            answer_type = "extract"
        top_entity = self._normalize_query_entity(data.get("entity", ""))

        query_state = {
            "entity": top_entity,
            "period": str(data.get("period", "") or "").strip(),
            "metric": str(data.get("metric", "") or "").strip(),
            "source_anchor": self._normalize_source_anchor(data.get("source_anchor")),
            "answer_type": answer_type,
            "required_slots": data.get("required_slots", []) if isinstance(data.get("required_slots"), list) else [],
            "unit": self._normalize_nullable_str(data.get("unit")),
            "rounding": self._normalize_nullable_str(data.get("rounding")),
            "missing_data_policy": normalize_missing_data_policy(data.get("missing_data_policy")),
        }
        sanitized_slots: list[Any] = []
        for slot in self._required_slots(query_state):
            struct = self._parse_slot_struct(slot)
            if not struct:
                sanitized_slots.append(slot)
                continue
            slot_entity = self._normalize_query_entity(struct.get("entity", ""))
            if slot_entity:
                struct["entity"] = slot_entity
            else:
                struct.pop("entity", None)
            sanitized_slots.append(struct)

        if not top_entity:
            for slot in sanitized_slots:
                struct = self._parse_slot_struct(slot)
                if not struct:
                    continue
                slot_entity = self._normalize_query_entity(struct.get("entity", ""))
                if slot_entity:
                    top_entity = slot_entity
                    break
            query_state["entity"] = top_entity

        aligned_slots: list[Any] = []
        for slot in sanitized_slots:
            struct = self._parse_slot_struct(slot)
            if not struct:
                aligned_slots.append(slot)
                continue
            slot_entity = self._normalize_query_entity(struct.get("entity", ""))
            if not slot_entity and top_entity:
                struct["entity"] = top_entity
            aligned_slots.append(struct)
        query_state["required_slots"] = aligned_slots

        if (
            query_state["answer_type"] == "compute"
            and query_state["missing_data_policy"] == "inapplicable_explain"
        ):
            query_state["missing_data_policy"] = "insufficient"

        return query_state

    def _raw_query_state_schema_errors(self, raw_data: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(raw_data, dict):
            errors.append("top-level must be JSON object")
            return errors
        entity_raw = str(raw_data.get("entity", "") or "").strip()
        if entity_raw and self._is_generic_entity_label(entity_raw):
            errors.append("entity must be specific or empty string")
        if not isinstance(raw_data.get("required_slots", None), list):
            errors.append("required_slots must be JSON array")
        else:
            for idx, slot in enumerate(raw_data.get("required_slots", [])):
                if not isinstance(slot, dict):
                    continue
                slot_entity_raw = str(slot.get("entity", "") or "").strip()
                if slot_entity_raw and self._is_generic_entity_label(slot_entity_raw):
                    errors.append(f"required_slots[{idx}].entity must be specific or empty string")
        policy_raw = raw_data.get("missing_data_policy", "insufficient")
        policy_text = str(policy_raw or "").strip().lower()
        if policy_text and policy_text not in self._VALID_MISSING_DATA_POLICIES:
            errors.append("missing_data_policy must be insufficient|zero_if_not_explicit|inapplicable_explain")
        answer_type = str(raw_data.get("answer_type", "") or "").strip().lower()
        if answer_type == "compute" and policy_text == "inapplicable_explain":
            errors.append("compute query must not set missing_data_policy=inapplicable_explain")
        return errors

    def _split_slot_structs(self, required_slots: list[Any]) -> tuple[list[dict[str, str]], int]:
        slot_structs: list[dict[str, str]] = []
        malformed_slots = 0
        for slot in required_slots:
            struct = self._parse_slot_struct(slot)
            metric = self._normalize_metric_text(struct.get("metric", "")) if struct else ""
            if not struct or not metric:
                malformed_slots += 1
                continue
            slot_structs.append(struct)
        return slot_structs, malformed_slots

    def _is_multi_period_compute_query(self, query: str, period: str) -> tuple[bool, list[str], list[str]]:
        query_years = sorted(set(self._extract_year_tokens(query)))
        period_years = sorted(set(self._extract_year_tokens(period)))
        query_lower = str(query or "").lower()
        multi_period = (
            len(query_years) >= 2
            or len(period_years) >= 2
            or "year-over-year" in query_lower
            or "yoy" in query_lower
            or " from " in query_lower and " to " in query_lower
            or " vs " in query_lower
            or " versus " in query_lower
            or " between " in query_lower
        )
        return multi_period, query_years, period_years

    def _compute_query_state_errors(
        self,
        query: str,
        query_state: dict[str, Any],
        required_slots: list[Any],
        slot_structs: list[dict[str, str]],
        malformed_slots: int,
    ) -> list[str]:
        errors: list[str] = []
        if not required_slots:
            errors.append("compute requires non-empty required_slots")
        if malformed_slots:
            errors.append("compute required_slots must be atomic slot objects")

        multi_period, query_years, period_years = self._is_multi_period_compute_query(
            query,
            str(query_state.get("period", "") or ""),
        )
        if multi_period and len(slot_structs) < 2:
            errors.append("multi-period compute needs >=2 slots")
        if (query_years or period_years) and slot_structs:
            has_period_per_slot = any(
                self._extract_year_tokens(str(slot.get("period", "") or ""))
                for slot in slot_structs
            )
            if not has_period_per_slot:
                errors.append("compute slots must carry explicit period")
        return errors

    @staticmethod
    def _dedupe_errors(errors: list[str]) -> list[str]:
        deduped: list[str] = []
        seen = set()
        for err in errors:
            key = err.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(err)
        return deduped

    def _explicit_anchor_errors(
        self,
        query: str,
        query_state: dict[str, Any],
        required_slots: list[Any],
        slot_structs: list[dict[str, str]],
    ) -> list[str]:
        if not self._query_has_explicit_statement_anchor(query):
            return []
        if not required_slots:
            return []

        errors: list[str] = []
        top_anchor = str(query_state.get("source_anchor", "") or "").strip().lower()
        slot_anchor_count = sum(
            1
            for slot in slot_structs
            if str(slot.get("source_anchor", "") or "").strip().lower()
        )
        if not top_anchor and slot_anchor_count == 0:
            errors.append("explicit statement query requires source_anchor at top-level or slots")
        if slot_structs and slot_anchor_count < len(slot_structs):
            errors.append("explicit statement query requires source_anchor in every required slot")
        return errors

    def _query_state_validation_errors(
        self,
        query: str,
        query_state: dict[str, Any],
        raw_data: Optional[Any] = None,
    ) -> list[str]:
        errors: list[str] = []
        if raw_data is not None:
            errors.extend(self._raw_query_state_schema_errors(raw_data))

        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        policy = normalize_missing_data_policy(query_state.get("missing_data_policy"))
        if policy not in self._VALID_MISSING_DATA_POLICIES:
            errors.append("missing_data_policy is invalid")
        if answer_type == "compute" and policy == "inapplicable_explain":
            errors.append("compute query must not set missing_data_policy=inapplicable_explain")

        required_slots = self._required_slots(query_state)
        slot_structs, malformed_slots = self._split_slot_structs(required_slots)

        if answer_type == "compute":
            errors.extend(
                self._compute_query_state_errors(
                    query=query,
                    query_state=query_state,
                    required_slots=required_slots,
                    slot_structs=slot_structs,
                    malformed_slots=malformed_slots,
                )
            )

        errors.extend(
            self._explicit_anchor_errors(
                query=query,
                query_state=query_state,
                required_slots=required_slots,
                slot_structs=slot_structs,
            )
        )

        return self._dedupe_errors(errors)

    def _query_state_retry_message(
        self,
        query: str,
        failed_output: Any,
        errors: list[str],
    ) -> str:
        reasons = "; ".join(errors[:4]) if errors else "schema violation"
        prev = self._compact_json(failed_output, max_chars=900)
        return QUERY_STATE_RETRY_PROMPT.format(
            errors=reasons,
            query=query,
            previous_output=prev,
        )

    async def _generate_query_state_with_retries(
        self,
        query: str,
        base_messages: list[dict[str, str]],
        initial_best_effort: dict[str, Any],
        *,
        log_label: str,
        exception_fallback: dict[str, Any],
    ) -> dict[str, Any]:
        max_attempts = 3
        baseline = self._sanitize_query_state(initial_best_effort or {})

        def validate(data: dict[str, Any]) -> tuple[bool, str]:
            sanitized = self._sanitize_query_state(data)
            errors = self._query_state_validation_errors(query, sanitized, raw_data=data)
            if errors:
                return False, "; ".join(errors)
            return True, ""

        def retry_message(data: dict[str, Any], reason: str) -> str:
            return self._query_state_retry_message(
                query=query,
                failed_output=data,
                errors=[reason] if reason else [],
            )

        try:
            data, ok, attempts = await generate_json_with_retries(
                self.llm,
                base_messages,
                validate,
                retry_message,
                max_attempts=max_attempts,
                logger=logger,
                warning_prefix=f"{log_label} json generation failed",
                model=self.stage_model,
            )
            if ok:
                return self._sanitize_query_state(data)
            if attempts:
                logger.warning("%s failed schema validation after %d attempts", log_label, len(attempts))
        except Exception as exc:
            logger.warning("%s failed: %s", log_label, exc)
            return exception_fallback
        if baseline:
            return baseline
        return self._sanitize_query_state(exception_fallback)

    async def _review_query_state(self, query: str, query_state: dict[str, Any]) -> dict[str, Any]:
        base_messages = [
            {
                "role": "user",
                "content": self._query_state_review_prompt_template().format(
                    query=query,
                    draft_query_state=self._compact_json(query_state, max_chars=1000),
                ),
            },
            {"role": "user", "content": QUERY_STATE_FORMAT_INSTRUCTION},
        ]
        return await self._generate_query_state_with_retries(
            query=query,
            base_messages=base_messages,
            initial_best_effort=query_state,
            log_label="Query state review",
            exception_fallback=query_state,
        )

    async def _infer_query_state(self, query: str) -> dict[str, Any]:
        base_messages = [
            {"role": "user", "content": self._query_state_prompt_template().format(query=query)},
            {"role": "user", "content": QUERY_STATE_FORMAT_INSTRUCTION},
        ]
        return await self._generate_query_state_with_retries(
            query=query,
            base_messages=base_messages,
            initial_best_effort={},
            log_label="Query state inference",
            exception_fallback={},
        )

    async def _initialize_query_state_phase(self, state: AgentState) -> None:
        started = time.perf_counter()
        state.query_state = await self._infer_query_state(state.user_query)
        state.query_state = await self._review_query_state(state.user_query, state.query_state)
        state.query_state = self._apply_query_state_heuristics(state.user_query, state.query_state)
        state.missing_slots = self._resolve_missing_slots(
            state.query_state,
            state.evidence_ledger,
            model_missing_slots=None,
        )
        append_trace(
            state.trace,
            step="query_state",
            input=state.user_query,
            output=state.query_state,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
