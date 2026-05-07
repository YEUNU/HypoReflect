"""Planning artifacts maintained inside the Execution loop:
QueryState init/sanitize/review (paper §3.2.2 → execution-time refresh),
required_slots tracking, and entity matching."""
from typing import Any
from typing import Any, Optional
import logging
import re
import time
from models.hyporeflect.stages.common import normalize_missing_data_policy
from models.hyporeflect.stages.llm_json import generate_json_with_retries
from models.hyporeflect.state import AgentState
from models.hyporeflect.trace import append_trace
from utils.prompts import QUERY_STATE_FORMAT_INSTRUCTION, QUERY_STATE_RETRY_PROMPT
logger = logging.getLogger(__name__)

class QueryStateSupport:

    def _apply_query_state_heuristics(self, query: str, query_state: dict[str, Any]) -> dict[str, Any]:
        query_lower = str(query or '').strip().lower()
        if not query_lower:
            return query_state
        adjusted = dict(query_state or {})
        answer_type = str(adjusted.get('answer_type', '') or '').strip().lower()
        if answer_type == 'extract':
            if re.match('^(is|are|was|were|does|do|did|has|have|can|could|should|would)\\b', query_lower):
                adjusted['answer_type'] = 'boolean'
            elif re.match('^(which|what are|list|name)\\b', query_lower):
                adjusted['answer_type'] = 'list'
        answer_type = str(adjusted.get('answer_type', '') or '').strip().lower()
        metric_text = str(adjusted.get('metric', '') or '').strip()
        metric_key = self._canonical_metric_key(metric_text)
        has_explicit_anchor = self._query_has_explicit_statement_anchor(query_lower)
        explicit_anchors = self._extract_query_statement_anchors(query_lower)
        if has_explicit_anchor and explicit_anchors:
            current_anchor = self._normalize_source_anchor(adjusted.get('source_anchor'))
            if not current_anchor:
                if len(explicit_anchors) == 1:
                    current_anchor = explicit_anchors[0]
                else:
                    inferred_top = self._infer_anchor_for_metric(metric_text)
                    if inferred_top in explicit_anchors:
                        current_anchor = inferred_top
            if current_anchor:
                adjusted['source_anchor'] = current_anchor
            slots_raw = adjusted.get('required_slots', [])
            if isinstance(slots_raw, list) and slots_raw:
                backfilled_slots: list[Any] = []
                for slot in slots_raw:
                    struct = self._parse_slot_struct(slot)
                    if not struct:
                        backfilled_slots.append(slot)
                        continue
                    slot_anchor = self._normalize_source_anchor(struct.get('source_anchor'))
                    if slot_anchor:
                        backfilled_slots.append(struct)
                        continue
                    inferred_slot_anchor = self._infer_anchor_for_metric(struct.get('metric', ''))
                    if inferred_slot_anchor and inferred_slot_anchor in explicit_anchors:
                        struct['source_anchor'] = inferred_slot_anchor
                    elif current_anchor:
                        struct['source_anchor'] = current_anchor
                    backfilled_slots.append(struct)
                adjusted['required_slots'] = backfilled_slots
        if not has_explicit_anchor and answer_type in {'extract', 'boolean', 'list'} and ('dividend' in metric_key):
            if self._normalize_source_anchor(adjusted.get('source_anchor')) == 'cash flow statement':
                adjusted['source_anchor'] = None
            slots = adjusted.get('required_slots', [])
            if isinstance(slots, list):
                relaxed_slots: list[Any] = []
                for slot in slots:
                    struct = self._parse_slot_struct(slot)
                    if not struct:
                        relaxed_slots.append(slot)
                        continue
                    if self._is_dividend_metric(struct.get('metric', '')) and str(struct.get('source_anchor', '') or '').strip().lower() == 'cash flow statement':
                        struct.pop('source_anchor', None)
                    relaxed_slots.append(struct)
                adjusted['required_slots'] = relaxed_slots
        if 'quick ratio' in metric_key or 'quick ratio' in query_lower:
            adjusted['metric'] = 'quick ratio'
            if re.match('^(is|are|was|were|does|do|did|has|have|can|could|should|would)\\b', query_lower):
                adjusted['answer_type'] = 'boolean'
            base_entity = str(adjusted.get('entity', '') or '').strip().lower()
            base_period = str(adjusted.get('period', '') or '').strip().lower()
            quick_slot: dict[str, Any] = {'metric': 'quick ratio'}
            if base_entity:
                quick_slot['entity'] = base_entity
            if base_period:
                quick_slot['period'] = base_period
            anchor = self._normalize_source_anchor(adjusted.get('source_anchor'))
            if has_explicit_anchor and anchor:
                quick_slot['source_anchor'] = anchor
                adjusted['source_anchor'] = anchor
            else:
                adjusted['source_anchor'] = None
            adjusted['required_slots'] = [quick_slot]
        if ('capital intensity' in metric_key or 'capital-intensive' in query_lower or 'capital intensive' in query_lower) and re.match('^(is|are|was|were|does|do|did|has|have|can|could|should|would)\\b', query_lower):
            adjusted['answer_type'] = 'boolean'
            adjusted['metric'] = 'capital intensity'
            base_entity = str(adjusted.get('entity', '') or '').strip().lower()
            base_period = str(adjusted.get('period', '') or '').strip().lower()
            capex_slot: dict[str, Any] = {'metric': 'capital expenditures'}
            revenue_slot: dict[str, Any] = {'metric': 'revenue'}
            if base_entity:
                capex_slot['entity'] = base_entity
                revenue_slot['entity'] = base_entity
            if base_period:
                capex_slot['period'] = base_period
                revenue_slot['period'] = base_period
            capex_slot['source_anchor'] = 'cash flow statement'
            revenue_slot['source_anchor'] = 'income statement'
            adjusted['source_anchor'] = None
            adjusted['required_slots'] = [capex_slot, revenue_slot]
        driver_intent = any((marker in query_lower for marker in ['what drove', 'what drives', 'driven by', 'driver', 'drivers', 'caused by']))
        if driver_intent and answer_type in {'extract', 'list'} and metric_text:
            slot_source_anchor = self._normalize_source_anchor(adjusted.get('source_anchor'))
            driver_metric = f'{metric_text} change drivers'
            slots_raw = adjusted.get('required_slots', [])
            slots = list(slots_raw) if isinstance(slots_raw, list) else []
            has_driver_slot = False
            for slot in slots:
                struct = self._parse_slot_struct(slot)
                if struct and 'driver' in self._canonical_metric_key(struct.get('metric', '')) and self._metric_matches(struct.get('metric', ''), driver_metric):
                    has_driver_slot = True
                    break
            if not has_driver_slot:
                driver_slot: dict[str, Any] = {'entity': str(adjusted.get('entity', '') or '').strip().lower(), 'period': str(adjusted.get('period', '') or '').strip().lower(), 'metric': driver_metric.lower()}
                if slot_source_anchor:
                    driver_slot['source_anchor'] = slot_source_anchor
                slots.append(driver_slot)
                adjusted['required_slots'] = slots
        if 'exclude' in query_lower and any((tok in query_lower for tok in ['m&a', 'acquisition', 'divestiture'])) and ('segment' in query_lower) and any((tok in metric_key for tok in ['segment growth', 'growth impact'])):
            adjusted_metric = 'organic sales change by business segment excluding acquisitions/divestitures'
            adjusted['metric'] = adjusted_metric
            slots_raw = adjusted.get('required_slots', [])
            if isinstance(slots_raw, list):
                rewritten_slots: list[Any] = []
                for slot in slots_raw:
                    struct = self._parse_slot_struct(slot)
                    if not struct:
                        rewritten_slots.append(slot)
                        continue
                    if any((tok in self._canonical_metric_key(struct.get('metric', '')) for tok in ['segment growth', 'growth impact'])):
                        struct['metric'] = adjusted_metric.lower()
                    rewritten_slots.append(struct)
                adjusted['required_slots'] = rewritten_slots
        return adjusted

    def _sanitize_query_state(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        answer_type = str(data.get('answer_type', 'extract')).strip().lower()
        if answer_type not in self._VALID_ANSWER_TYPES:
            answer_type = 'extract'
        top_entity = self._normalize_query_entity(data.get('entity', ''))
        query_state = {'entity': top_entity, 'period': str(data.get('period', '') or '').strip(), 'metric': str(data.get('metric', '') or '').strip(), 'source_anchor': self._normalize_source_anchor(data.get('source_anchor')), 'answer_type': answer_type, 'required_slots': data.get('required_slots', []) if isinstance(data.get('required_slots'), list) else [], 'unit': self._normalize_nullable_str(data.get('unit')), 'rounding': self._normalize_nullable_str(data.get('rounding')), 'missing_data_policy': normalize_missing_data_policy(data.get('missing_data_policy'))}
        sanitized_slots: list[Any] = []
        for slot in self._required_slots(query_state):
            struct = self._parse_slot_struct(slot)
            if not struct:
                sanitized_slots.append(slot)
                continue
            slot_entity = self._normalize_query_entity(struct.get('entity', ''))
            if slot_entity:
                struct['entity'] = slot_entity
            else:
                struct.pop('entity', None)
            sanitized_slots.append(struct)
        if not top_entity:
            for slot in sanitized_slots:
                struct = self._parse_slot_struct(slot)
                if not struct:
                    continue
                slot_entity = self._normalize_query_entity(struct.get('entity', ''))
                if slot_entity:
                    top_entity = slot_entity
                    break
            query_state['entity'] = top_entity
        aligned_slots: list[Any] = []
        for slot in sanitized_slots:
            struct = self._parse_slot_struct(slot)
            if not struct:
                aligned_slots.append(slot)
                continue
            slot_entity = self._normalize_query_entity(struct.get('entity', ''))
            if not slot_entity and top_entity:
                struct['entity'] = top_entity
            aligned_slots.append(struct)
        query_state['required_slots'] = aligned_slots
        if query_state['answer_type'] == 'compute' and query_state['missing_data_policy'] == 'inapplicable_explain':
            query_state['missing_data_policy'] = 'insufficient'
        return query_state

    def _raw_query_state_schema_errors(self, raw_data: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(raw_data, dict):
            errors.append('top-level must be JSON object')
            return errors
        entity_raw = str(raw_data.get('entity', '') or '').strip()
        if entity_raw and self._is_generic_entity_label(entity_raw):
            errors.append('entity must be specific or empty string')
        if not isinstance(raw_data.get('required_slots', None), list):
            errors.append('required_slots must be JSON array')
        else:
            for idx, slot in enumerate(raw_data.get('required_slots', [])):
                if not isinstance(slot, dict):
                    continue
                slot_entity_raw = str(slot.get('entity', '') or '').strip()
                if slot_entity_raw and self._is_generic_entity_label(slot_entity_raw):
                    errors.append(f'required_slots[{idx}].entity must be specific or empty string')
        policy_raw = raw_data.get('missing_data_policy', 'insufficient')
        policy_text = str(policy_raw or '').strip().lower()
        if policy_text and policy_text not in self._VALID_MISSING_DATA_POLICIES:
            errors.append('missing_data_policy must be insufficient|zero_if_not_explicit|inapplicable_explain')
        answer_type = str(raw_data.get('answer_type', '') or '').strip().lower()
        if answer_type == 'compute' and policy_text == 'inapplicable_explain':
            errors.append('compute query must not set missing_data_policy=inapplicable_explain')
        return errors

    def _split_slot_structs(self, required_slots: list[Any]) -> tuple[list[dict[str, str]], int]:
        slot_structs: list[dict[str, str]] = []
        malformed_slots = 0
        for slot in required_slots:
            struct = self._parse_slot_struct(slot)
            metric = self._normalize_metric_text(struct.get('metric', '')) if struct else ''
            if not struct or not metric:
                malformed_slots += 1
                continue
            slot_structs.append(struct)
        return (slot_structs, malformed_slots)

    def _is_multi_period_compute_query(self, query: str, period: str) -> tuple[bool, list[str], list[str]]:
        query_years = sorted(set(self._extract_year_tokens(query)))
        period_years = sorted(set(self._extract_year_tokens(period)))
        query_lower = str(query or '').lower()
        multi_period = len(query_years) >= 2 or len(period_years) >= 2 or 'year-over-year' in query_lower or ('yoy' in query_lower) or (' from ' in query_lower and ' to ' in query_lower) or (' vs ' in query_lower) or (' versus ' in query_lower) or (' between ' in query_lower)
        return (multi_period, query_years, period_years)

    def _compute_query_state_errors(self, query: str, query_state: dict[str, Any], required_slots: list[Any], slot_structs: list[dict[str, str]], malformed_slots: int) -> list[str]:
        errors: list[str] = []
        if not required_slots:
            errors.append('compute requires non-empty required_slots')
        if malformed_slots:
            errors.append('compute required_slots must be atomic slot objects')
        multi_period, query_years, period_years = self._is_multi_period_compute_query(query, str(query_state.get('period', '') or ''))
        if multi_period and len(slot_structs) < 2:
            errors.append('multi-period compute needs >=2 slots')
        if (query_years or period_years) and slot_structs:
            has_period_per_slot = any((self._extract_year_tokens(str(slot.get('period', '') or '')) for slot in slot_structs))
            if not has_period_per_slot:
                errors.append('compute slots must carry explicit period')
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

    def _explicit_anchor_errors(self, query: str, query_state: dict[str, Any], required_slots: list[Any], slot_structs: list[dict[str, str]]) -> list[str]:
        if not self._query_has_explicit_statement_anchor(query):
            return []
        if not required_slots:
            return []
        errors: list[str] = []
        top_anchor = str(query_state.get('source_anchor', '') or '').strip().lower()
        slot_anchor_count = sum((1 for slot in slot_structs if str(slot.get('source_anchor', '') or '').strip().lower()))
        if not top_anchor and slot_anchor_count == 0:
            errors.append('explicit statement query requires source_anchor at top-level or slots')
        if slot_structs and slot_anchor_count < len(slot_structs):
            errors.append('explicit statement query requires source_anchor in every required slot')
        return errors

    def _query_state_validation_errors(self, query: str, query_state: dict[str, Any], raw_data: Optional[Any]=None) -> list[str]:
        errors: list[str] = []
        if raw_data is not None:
            errors.extend(self._raw_query_state_schema_errors(raw_data))
        answer_type = str(query_state.get('answer_type', '') or '').strip().lower()
        policy = normalize_missing_data_policy(query_state.get('missing_data_policy'))
        if policy not in self._VALID_MISSING_DATA_POLICIES:
            errors.append('missing_data_policy is invalid')
        if answer_type == 'compute' and policy == 'inapplicable_explain':
            errors.append('compute query must not set missing_data_policy=inapplicable_explain')
        required_slots = self._required_slots(query_state)
        slot_structs, malformed_slots = self._split_slot_structs(required_slots)
        if answer_type == 'compute':
            errors.extend(self._compute_query_state_errors(query=query, query_state=query_state, required_slots=required_slots, slot_structs=slot_structs, malformed_slots=malformed_slots))
        errors.extend(self._explicit_anchor_errors(query=query, query_state=query_state, required_slots=required_slots, slot_structs=slot_structs))
        return self._dedupe_errors(errors)

    def _query_state_retry_message(self, query: str, failed_output: Any, errors: list[str]) -> str:
        reasons = '; '.join(errors[:4]) if errors else 'schema violation'
        prev = self._compact_json(failed_output, max_chars=900)
        return QUERY_STATE_RETRY_PROMPT.format(errors=reasons, query=query, previous_output=prev)

    async def _generate_query_state_with_retries(self, query: str, base_messages: list[dict[str, str]], initial_best_effort: dict[str, Any], *, log_label: str, exception_fallback: dict[str, Any]) -> dict[str, Any]:
        max_attempts = 3
        baseline = self._sanitize_query_state(initial_best_effort or {})

        def validate(data: dict[str, Any]) -> tuple[bool, str]:
            sanitized = self._sanitize_query_state(data)
            errors = self._query_state_validation_errors(query, sanitized, raw_data=data)
            if errors:
                return (False, '; '.join(errors))
            return (True, '')

        def retry_message(data: dict[str, Any], reason: str) -> str:
            return self._query_state_retry_message(query=query, failed_output=data, errors=[reason] if reason else [])
        try:
            data, ok, attempts = await generate_json_with_retries(self.llm, base_messages, validate, retry_message, max_attempts=max_attempts, logger=logger, warning_prefix=f'{log_label} json generation failed', model=self.stage_model)
            if ok:
                return self._sanitize_query_state(data)
            if attempts:
                logger.warning('%s failed schema validation after %d attempts', log_label, len(attempts))
        except Exception as exc:
            logger.warning('%s failed: %s', log_label, exc)
            return exception_fallback
        if baseline:
            return baseline
        return self._sanitize_query_state(exception_fallback)

    async def _review_query_state(self, query: str, query_state: dict[str, Any]) -> dict[str, Any]:
        base_messages = [{'role': 'user', 'content': self._query_state_review_prompt_template().format(query=query, draft_query_state=self._compact_json(query_state, max_chars=1000))}, {'role': 'user', 'content': QUERY_STATE_FORMAT_INSTRUCTION}]
        return await self._generate_query_state_with_retries(query=query, base_messages=base_messages, initial_best_effort=query_state, log_label='Query state review', exception_fallback=query_state)

    async def _infer_query_state(self, query: str) -> dict[str, Any]:
        base_messages = [{'role': 'user', 'content': self._query_state_prompt_template().format(query=query)}, {'role': 'user', 'content': QUERY_STATE_FORMAT_INSTRUCTION}]
        return await self._generate_query_state_with_retries(query=query, base_messages=base_messages, initial_best_effort={}, log_label='Query state inference', exception_fallback={})

    async def _initialize_query_state_phase(self, state: AgentState) -> None:
        started = time.perf_counter()
        state.query_state = await self._infer_query_state(state.user_query)
        state.query_state = await self._review_query_state(state.user_query, state.query_state)
        state.query_state = self._apply_query_state_heuristics(state.user_query, state.query_state)
        state.missing_slots = self._resolve_missing_slots(state.query_state, state.evidence_ledger, model_missing_slots=None)
        append_trace(state.trace, step='query_state', input=state.user_query, output=state.query_state, duration_ms=(time.perf_counter() - started) * 1000.0)

class SlotSupport:

    @staticmethod
    def _normalize_source_anchor(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text or text in {'null', 'none', 'n/a'}:
            return None
        has_income = any((tok in text for tok in ['income statement', 'statement of income', 'p&l', 'profit and loss']))
        has_balance = any((tok in text for tok in ['balance sheet', 'statement of financial position']))
        has_cashflow = any((tok in text for tok in ['cash flow statement', 'statement of cash flows', 'cash flow']))
        has_note = 'note table' in text
        hit_count = sum((1 for flag in [has_income, has_balance, has_cashflow, has_note] if flag))
        if hit_count > 1:
            return None
        if has_income:
            return 'income statement'
        if has_balance:
            return 'balance sheet'
        if has_cashflow:
            return 'cash flow statement'
        if has_note:
            return 'note table'
        return None

    @staticmethod
    def _parse_slot_struct(value: Any) -> Optional[dict[str, str]]:
        if not isinstance(value, dict):
            return None
        raw = value
        normalized: dict[str, str] = {}
        for key in ('entity', 'period', 'metric', 'source_anchor', 'unit', 'rounding'):
            if key not in raw:
                continue
            val = raw.get(key)
            if val is None:
                continue
            if key == 'source_anchor':
                anchor = SlotSupport._normalize_source_anchor(val)
                if anchor:
                    normalized[key] = anchor
                continue
            text = re.sub('\\s+', ' ', str(val).strip().lower())
            if not text or text in {'none', 'null', 'n/a'}:
                continue
            normalized[key] = text
        return normalized or None

    @staticmethod
    def _normalize_slot(value: Any) -> str:
        struct = SlotSupport._parse_slot_struct(value)
        if struct:
            ordered_keys = ('entity', 'period', 'metric', 'source_anchor', 'unit', 'rounding')
            return '|'.join((f'{key}={struct[key]}' for key in ordered_keys if key in struct))
        return re.sub('\\s+', ' ', str(value or '').strip().lower())

    def _normalize_slot_conflict_strategy(self, value: Any) -> str:
        strategy = str(value or 'best_supported').strip().lower()
        if strategy in self._VALID_SLOT_CONFLICT_STRATEGIES:
            return strategy
        return 'best_supported'

    def _ledger_value_conflict_key(self, value: Any) -> str:
        text = re.sub('\\s+', ' ', str(value or '').strip().lower())
        if not text:
            return ''
        numeric_candidates = self._extract_scaled_numeric_candidates(text)
        if numeric_candidates:
            sig = max(numeric_candidates, key=lambda num: abs(num))
            return f'num:{sig:.12g}'
        return f'txt:{text}'

    def _required_slots(self, query_state: dict[str, Any]) -> list[Any]:
        slots_raw = query_state.get('required_slots', [])
        slots: list[Any] = []
        if isinstance(slots_raw, list):
            for item in slots_raw:
                struct = self._parse_slot_struct(item)
                if struct:
                    slots.append(struct)
                    continue
                text = str(item or '').strip()
                if text:
                    slots.append(text)
        deduped: list[Any] = []
        seen = set()
        for slot in slots:
            key = self._normalize_slot(slot)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(slot)
        return deduped

    def _compute_missing_slots(self, query_state: dict[str, Any], evidence_ledger: list[dict[str, str]], slot_conflict_strategy: Optional[str]=None) -> list[Any]:
        required = self._required_slots(query_state)
        required_map = {self._normalize_slot(slot): slot for slot in required}
        values_by_slot: dict[str, set[str]] = {key: set() for key in required_map}
        conflict_strategy = self._normalize_slot_conflict_strategy(slot_conflict_strategy)
        for item in evidence_ledger:
            slot_key = self._normalize_slot(item.get('slot', ''))
            if slot_key not in required_map:
                continue
            value_norm = re.sub('\\s+', ' ', str(item.get('value', '') or '').strip().lower())
            if value_norm in {'missing', 'n/a', 'na', 'none', 'null', 'unknown', '-'}:
                continue
            if value_norm:
                values_by_slot[slot_key].add(self._ledger_value_conflict_key(value_norm))
        missing: list[Any] = []
        for key, slot in required_map.items():
            values = values_by_slot.get(key, set())
            if len(values) == 0:
                missing.append(slot)
                continue
            if len(values) > 1 and conflict_strategy == 'keep_missing_on_tie':
                missing.append(slot)
        return missing

    def _slot_period_group_key(self, slot_struct: dict[str, str]) -> str:
        entity = re.sub('\\s+', ' ', str(slot_struct.get('entity', '') or '').strip().lower())
        metric = self._canonical_metric_key(slot_struct.get('metric', ''))
        anchor = self._normalize_source_anchor(slot_struct.get('source_anchor')) or ''
        if not metric:
            return ''
        return f'{entity}|{metric}|{anchor}'

    def _collapsed_multi_period_slots(self, query_state: dict[str, Any], evidence_ledger: list[dict[str, str]]) -> list[Any]:
        required = self._required_slots(query_state)
        if not required:
            return []
        required_map = {self._normalize_slot(slot): slot for slot in required}
        values_by_slot: dict[str, set[str]] = {key: set() for key in required_map}
        for item in evidence_ledger:
            slot_key = self._normalize_slot(item.get('slot', ''))
            if slot_key not in required_map:
                continue
            value_norm = re.sub('\\s+', ' ', str(item.get('value', '') or '').strip().lower())
            if not value_norm:
                continue
            values_by_slot[slot_key].add(self._ledger_value_conflict_key(value_norm))
        slots_by_group: dict[str, list[str]] = {}
        for slot_key, slot_raw in required_map.items():
            struct = self._parse_slot_struct(slot_raw)
            if not struct:
                continue
            years = set(self._extract_year_tokens(struct.get('period', '')))
            if not years:
                continue
            group_key = self._slot_period_group_key(struct)
            if not group_key:
                continue
            slots_by_group.setdefault(group_key, []).append(slot_key)
        collapsed: list[Any] = []
        seen_slot_keys = set()
        for slot_keys in slots_by_group.values():
            if len(slot_keys) < 2:
                continue
            period_signatures = set()
            value_keys: list[str] = []
            complete = True
            for slot_key in slot_keys:
                slot_struct = self._parse_slot_struct(required_map.get(slot_key, ''))
                period_years = tuple(sorted(set(self._extract_year_tokens(str((slot_struct or {}).get('period', ''))))))
                if period_years:
                    period_signatures.add(period_years)
                values = values_by_slot.get(slot_key, set())
                if not values:
                    complete = False
                    break
                value_keys.append(sorted(values)[0])
            if not complete or len(period_signatures) < 2:
                continue
            if len(set(value_keys)) > 1:
                continue
            for slot_key in slot_keys:
                if slot_key in seen_slot_keys:
                    continue
                seen_slot_keys.add(slot_key)
                collapsed.append(required_map[slot_key])
        return collapsed

    def _sanitize_missing_slots(self, query_state: dict[str, Any], missing_slots_raw: Any) -> Optional[list[Any]]:
        required = self._required_slots(query_state)
        if not required:
            return []
        if not isinstance(missing_slots_raw, list):
            return None
        required_map = {self._normalize_slot(slot): slot for slot in required}
        seen = set()
        raw_nonempty = False
        for item in missing_slots_raw:
            key = self._normalize_slot(item)
            if not key:
                continue
            raw_nonempty = True
            if key in required_map:
                seen.add(key)
        if raw_nonempty and (not seen):
            return None
        return [slot for slot in required if self._normalize_slot(slot) in seen]

    def _resolve_missing_slots(self, query_state: dict[str, Any], evidence_ledger: list[dict[str, str]], model_missing_slots: Optional[list[Any]], trust_model_missing: bool=False) -> list[Any]:
        required = self._required_slots(query_state)
        if not required:
            return []
        ledger_missing = self._compute_missing_slots(query_state, evidence_ledger)
        if not trust_model_missing:
            return ledger_missing
        if model_missing_slots is None:
            return ledger_missing
        if not evidence_ledger and (not trust_model_missing):
            return required
        ledger_keys = {self._normalize_slot(slot) for slot in ledger_missing}
        model_keys = {self._normalize_slot(slot) for slot in model_missing_slots}
        target_keys = model_keys if trust_model_missing else ledger_keys | model_keys
        if not target_keys:
            return []
        return [slot for slot in required if self._normalize_slot(slot) in target_keys]

    def _slot_struct_matches(self, candidate: dict[str, str], required: dict[str, str]) -> bool:
        if not isinstance(candidate, dict) or not isinstance(required, dict):
            return False
        if not self._metric_matches(candidate.get('metric', ''), required.get('metric', '')):
            return False
        if not self._periods_overlap(candidate.get('period', ''), required.get('period', '')):
            return False
        if not self._entity_matches(candidate.get('entity', ''), required.get('entity', '')):
            return False
        required_anchor = str(required.get('source_anchor', '') or '').strip().lower()
        candidate_anchor = str(candidate.get('source_anchor', '') or '').strip().lower()
        if required_anchor and candidate_anchor and (required_anchor != candidate_anchor):
            return False
        return True

    def _slot_key_by_year_hint(self, *, compatible_keys: list[str], required_map: dict[str, Any], citation: str, citation_span: str, value: str) -> Optional[str]:
        if len(compatible_keys) <= 1:
            return compatible_keys[0] if compatible_keys else None
        slot_years: dict[str, set[str]] = {}
        for key in compatible_keys:
            req_struct = self._parse_slot_struct(required_map.get(key, ''))
            if not req_struct:
                continue
            years = set(self._extract_year_tokens(str(req_struct.get('period', '') or '')))
            if years:
                slot_years[key] = years
        if not slot_years:
            return None
        title_years = set(self._extract_year_tokens(self._citation_doc_title(citation)))
        if title_years:
            title_matches = [key for key, years in slot_years.items() if not years.isdisjoint(title_years)]
            if len(title_matches) == 1:
                return title_matches[0]
        value_nums = self._extract_scaled_numeric_candidates(str(value or ''))
        if not value_nums or not citation_span:
            return None
        span_text = str(citation_span or '')
        span_lower = span_text.lower()
        supported: list[str] = []
        for key, years in slot_years.items():
            matched = False
            for year in years:
                for match in re.finditer(re.escape(year), span_lower):
                    start = max(0, match.start() - 96)
                    end = min(len(span_text), match.end() + 96)
                    window = span_text[start:end]
                    window_nums = self._extract_scaled_numeric_candidates(window)
                    if not window_nums:
                        continue
                    for expected in value_nums:
                        tol = max(1e-06, abs(expected) * 0.0001)
                        if any((abs(found - expected) <= tol for found in window_nums)):
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                supported.append(key)
        if len(supported) == 1:
            return supported[0]
        return None

    def _resolve_required_slot_key(self, slot_raw: Any, required_map: dict[str, Any], query_state: dict[str, Any], *, value: str='', citation: str='', citation_span: str='') -> tuple[Optional[str], str]:
        direct_key = self._normalize_slot(slot_raw)
        if direct_key and direct_key in required_map:
            return (direct_key, 'exact_slot_match')
        slot_struct = self._parse_slot_struct(slot_raw)
        if not slot_struct:
            slot_text = str(slot_raw or '').strip()
            if not slot_text:
                return (None, 'missing_slot')
            canonical_slot = self._canonical_metric_key(slot_text)
            if not canonical_slot:
                return (None, 'slot_unmatched')
            matches: list[str] = []
            for req_key, req_slot in required_map.items():
                req_struct = self._parse_slot_struct(req_slot)
                req_metric = req_struct.get('metric', '') if req_struct else req_slot
                if self._metric_matches(canonical_slot, req_metric):
                    matches.append(req_key)
            if len(matches) == 1:
                return (matches[0], 'metric_fallback_match')
            if len(matches) > 1:
                return (matches[0], 'metric_fallback_ambiguous')
            return (None, 'slot_unmatched')
        query_entity = str(query_state.get('entity', '') or '').strip()
        compatible: list[str] = []
        mismatch_counts: dict[str, int] = {}
        for req_key, req_slot in required_map.items():
            req_struct = self._parse_slot_struct(req_slot)
            if not req_struct:
                continue
            if not self._metric_matches(slot_struct.get('metric', ''), req_struct.get('metric', '')):
                mismatch_counts['metric_mismatch'] = mismatch_counts.get('metric_mismatch', 0) + 1
                continue
            if not self._periods_overlap(slot_struct.get('period', ''), req_struct.get('period', '')):
                mismatch_counts['period_mismatch'] = mismatch_counts.get('period_mismatch', 0) + 1
                continue
            req_entity = str(req_struct.get('entity', '') or '').strip()
            cand_entity = str(slot_struct.get('entity', '') or '').strip()
            if self._is_generic_entity_label(req_entity) and (not self._is_generic_entity_label(query_entity)):
                req_entity = query_entity
            if self._is_generic_entity_label(cand_entity):
                cand_entity = ''
            if req_entity and cand_entity and (not self._entity_matches(cand_entity, req_entity)):
                mismatch_counts['entity_mismatch'] = mismatch_counts.get('entity_mismatch', 0) + 1
                continue
            req_anchor = str(req_struct.get('source_anchor', '') or '').strip().lower()
            cand_anchor = str(slot_struct.get('source_anchor', '') or '').strip().lower()
            if req_anchor and cand_anchor and (req_anchor != cand_anchor):
                mismatch_counts['anchor_mismatch'] = mismatch_counts.get('anchor_mismatch', 0) + 1
                continue
            compatible.append(req_key)
        if len(compatible) == 1:
            return (compatible[0], 'structural_slot_match')
        if len(compatible) > 1:
            hinted = self._slot_key_by_year_hint(compatible_keys=compatible, required_map=required_map, citation=citation, citation_span=citation_span, value=value)
            if hinted:
                return (hinted, 'structural_slot_tiebreak_year_hint')
            compatible.sort(key=lambda key: len(self._normalize_slot(required_map[key])), reverse=True)
            return (compatible[0], 'structural_slot_tiebreak')
        if mismatch_counts:
            reason = max(mismatch_counts.items(), key=lambda item: item[1])[0]
            return (None, reason)
        return (None, 'slot_unmatched')

class EntitySupport:

    @staticmethod
    def _normalize_metric_text(value: Any) -> str:
        text = str(value or '').strip().lower()
        if not text:
            return ''
        text = text.replace('_', ' ')
        text = re.sub('\\s+', ' ', text)
        return text

    @staticmethod
    def _normalize_entity_key(value: Any) -> str:
        return re.sub('[^a-z0-9]+', '', str(value or '').strip().lower())

    def _entity_alias_keys(self, entity: Any) -> set[str]:
        raw = str(entity or '').strip().lower()
        key = self._normalize_entity_key(raw)
        if not key:
            return set()
        aliases: set[str] = {key}
        tokens = [tok for tok in re.split('[^a-z0-9]+', raw) if tok]
        if not tokens:
            return aliases
        legal_suffixes = {'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'plc', 'ltd', 'llc', 'lp', 'sa', 'ag', 'nv', 'group', 'holdings', 'holding', 'the'}
        core_tokens = [tok for tok in tokens if tok not in legal_suffixes]
        if core_tokens:
            aliases.add(''.join(core_tokens))
            for tok in core_tokens:
                if len(tok) >= 2:
                    aliases.add(tok)
        return aliases

    def _entity_search_aliases(self, entity: Any) -> list[str]:
        raw = str(entity or '').strip()
        if not raw or self._is_generic_entity_label(raw):
            return []
        aliases = []
        seen = set()

        def add(value: Any) -> None:
            text = str(value or '').strip()
            key = text.lower()
            if not key or key in seen:
                return
            seen.add(key)
            aliases.append(text)
        add(raw)
        normalized = self._normalize_entity_key(raw)
        if normalized:
            add(normalized)
        for token in self._entity_alias_keys(raw):
            if len(token) >= 4:
                add(token)
        return aliases

    def _query_entity_candidates(self, query_state: dict[str, Any], user_query: str='') -> list[str]:
        candidates: list[str] = []
        top_entity = str(query_state.get('entity', '') or '').strip()
        if top_entity and (not self._is_generic_entity_label(top_entity)):
            candidates.append(top_entity)
        for slot in self._required_slots(query_state):
            struct = self._parse_slot_struct(slot)
            if not struct:
                continue
            slot_entity = str(struct.get('entity', '') or '').strip()
            if slot_entity and (not self._is_generic_entity_label(slot_entity)):
                candidates.append(slot_entity)
        deduped: list[str] = []
        seen = set()
        for value in candidates:
            key = str(value or '').strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(value).strip())
        return deduped

    def _entity_matches(self, lhs: Any, rhs: Any) -> bool:
        left = self._entity_alias_keys(lhs)
        right = self._entity_alias_keys(rhs)
        if not left or not right:
            return True
        if not left.isdisjoint(right):
            return True
        for lval in left:
            for rval in right:
                if not lval or not rval:
                    continue
                if lval in rval or rval in lval:
                    return True
        return False

    def _canonical_metric_key(self, metric: Any) -> str:
        text = self._normalize_metric_text(metric)
        if not text:
            return ''
        text = re.sub('(?<!\\d)(?:fy\\s*)?(?:19|20)\\d{2}(?!\\d)', ' ', text, flags=re.IGNORECASE)
        text = re.sub('[^a-z0-9]+', ' ', text)
        return re.sub('\\s+', ' ', text).strip()

    def _metric_matches(self, lhs: Any, rhs: Any) -> bool:
        left = self._canonical_metric_key(lhs)
        right = self._canonical_metric_key(rhs)
        if not left or not right:
            return False
        if left == right:
            return True
        if left in right or right in left:
            return True
        left_terms = {term for term in self._metric_alias_terms(left) if len(term.strip()) >= 4}
        right_terms = {term for term in self._metric_alias_terms(right) if len(term.strip()) >= 4}
        if left_terms and right_terms and (not left_terms.isdisjoint(right_terms)):
            return True
        return False

    @staticmethod
    def _extract_quarter_tokens(text: str) -> list[str]:
        return re.findall('\\bq([1-4])\\b', str(text or '').lower())

    def _periods_overlap(self, lhs: Any, rhs: Any) -> bool:
        left = str(lhs or '').strip().lower()
        right = str(rhs or '').strip().lower()
        if not left or not right:
            return True
        left_years = set(self._extract_year_tokens(left))
        right_years = set(self._extract_year_tokens(right))
        if left_years and right_years and left_years.isdisjoint(right_years):
            return False
        left_quarters = set(self._extract_quarter_tokens(left))
        right_quarters = set(self._extract_quarter_tokens(right))
        if left_quarters and right_quarters:
            if (not left_years or not right_years or (not left_years.isdisjoint(right_years))) and left_quarters.isdisjoint(right_quarters):
                return False
        return True

    @staticmethod
    def _citation_doc_title(citation: str) -> str:
        match = re.search('^\\[\\[([^,\\]]+),\\s*Page\\s*\\d+\\s*,\\s*Chunk\\s*\\d+\\s*\\]\\]$', str(citation or '').strip(), flags=re.IGNORECASE)
        if not match:
            return ''
        return str(match.group(1) or '').strip()

    def _title_year_tokens(self, title: Any) -> list[str]:
        return self._extract_year_tokens(str(title or ''))

    def _filter_nodes_by_query_entity(self, nodes: list[dict[str, Any]], query_state: dict[str, Any], *, user_query: str='', fail_open: bool=False) -> list[dict[str, Any]]:
        if not nodes:
            return nodes
        entity_candidates = self._query_entity_candidates(query_state, user_query)
        if not entity_candidates:
            return nodes
        filtered: list[dict[str, Any]] = []
        for node in nodes:
            title = str(node.get('title') or node.get('doc') or '').strip()
            if not title:
                continue
            doc_target = title.split('_', 1)[0].strip()
            if not doc_target:
                continue
            if any((self._entity_matches(candidate, doc_target) for candidate in entity_candidates)):
                filtered.append(node)
        if filtered:
            metric_key = self._canonical_metric_key(query_state.get('metric', ''))
            debt_listing_query = any((marker in metric_key for marker in ['debt securities', 'registered to trade', 'national securities exchange', 'trading symbol']))
            if debt_listing_query:
                return filtered
            period_years = set(self._extract_year_tokens(str(query_state.get('period', '') or '')))
            if period_years:
                period_filtered: list[dict[str, Any]] = []
                for node in filtered:
                    title = str(node.get('title') or node.get('doc') or '').strip()
                    title_years = set(self._title_year_tokens(title))
                    if title_years and title_years.isdisjoint(period_years):
                        continue
                    period_filtered.append(node)
                if period_filtered:
                    return period_filtered
                if not fail_open:
                    return []
            return filtered
        return nodes if fail_open else []

    def _build_entity_retry_entities(self, query_state: dict[str, Any], user_query: str) -> list[str]:
        entity_candidates = self._query_entity_candidates(query_state, user_query)
        if not entity_candidates:
            return []
        period = str(query_state.get('period', '') or '').strip()
        metric = str(query_state.get('metric', '') or '').strip()
        candidates: list[str] = []
        for entity in entity_candidates:
            entity_aliases = self._entity_search_aliases(entity)
            if not entity_aliases:
                entity_aliases = [entity]
            candidates.extend(entity_aliases)
            for alias in entity_aliases[:2]:
                if period and metric:
                    candidates.append(f'{alias} {period} {metric}')
                elif metric:
                    candidates.append(f'{alias} {metric}')
                if user_query.strip():
                    candidates.append(f'{alias} {user_query.strip()}')
        deduped: list[str] = []
        seen = set()
        for item in candidates:
            key = str(item or '').strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(item).strip())
        return deduped

    async def _entity_guarded_graph_search(self, *, entities: list[str], depth: int, top_k: int, query_state: dict[str, Any], user_query: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        txt, data = await self._call_graph_search(entities, depth=depth, top_k=top_k)
        filtered = self._filter_nodes_by_query_entity(data, query_state, user_query=user_query, fail_open=False)
        diagnostics: dict[str, Any] = {'initial_raw': len(data), 'initial_kept': len(filtered), 'retry_used': False, 'retry_raw': 0, 'retry_kept': 0, 'relaxed_used': False, 'relaxed_kept': 0, 'retry_reason': ''}
        retry_data: list[dict[str, Any]] = []
        query_entity = str(query_state.get('entity', '') or '').strip()
        answer_type = str(query_state.get('answer_type', '') or '').strip().lower()
        period_years = set(self._extract_year_tokens(str(query_state.get('period', '') or '')))
        kept_years: set[str] = set()
        for node in filtered:
            title = str(node.get('title') or node.get('doc') or '').strip()
            if not title:
                continue
            kept_years.update(self._title_year_tokens(title))
        overlap_year_count = len(period_years & kept_years) if period_years else 0
        compute_sparse_year_coverage = answer_type == 'compute' and len(period_years) >= 2 and (overlap_year_count < min(2, len(period_years)))
        should_retry = diagnostics['initial_kept'] == 0 or compute_sparse_year_coverage
        if query_entity and (not self._is_generic_entity_label(query_entity)) and (diagnostics['initial_raw'] > 0) and should_retry:
            retry_entities = self._build_entity_retry_entities(query_state, user_query)
            if retry_entities:
                diagnostics['retry_used'] = True
                diagnostics['retry_reason'] = 'compute_sparse_year_coverage' if compute_sparse_year_coverage else 'initial_empty'
                retry_txt, retry_data = await self._call_graph_search(retry_entities, depth=depth, top_k=top_k)
                retry_filtered = self._filter_nodes_by_query_entity(retry_data, query_state, user_query=user_query, fail_open=False)
                diagnostics['retry_raw'] = len(retry_data)
                diagnostics['retry_kept'] = len(retry_filtered)
                if retry_txt:
                    txt = f'{txt}\n\n{retry_txt}' if txt else retry_txt
                if retry_filtered:
                    return (txt, retry_filtered, diagnostics)
        if not filtered:
            relaxed_source = retry_data if retry_data else data
            if relaxed_source:
                relaxed = self._filter_nodes_by_query_entity(relaxed_source, query_state, user_query=user_query, fail_open=True)
                if relaxed:
                    diagnostics['relaxed_used'] = True
                    diagnostics['relaxed_kept'] = len(relaxed)
                    return (txt, relaxed, diagnostics)
        return (txt, filtered, diagnostics)
