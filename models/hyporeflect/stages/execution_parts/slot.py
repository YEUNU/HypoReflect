import re
from typing import Any, Optional


class SlotSupport:
    @staticmethod
    def _normalize_source_anchor(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text or text in {"null", "none", "n/a"}:
            return None
        has_income = any(tok in text for tok in ["income statement", "statement of income", "p&l", "profit and loss"])
        has_balance = any(tok in text for tok in ["balance sheet", "statement of financial position"])
        has_cashflow = any(tok in text for tok in ["cash flow statement", "statement of cash flows", "cash flow"])
        has_note = "note table" in text
        hit_count = sum(1 for flag in [has_income, has_balance, has_cashflow, has_note] if flag)
        if hit_count > 1:
            return None
        if has_income:
            return "income statement"
        if has_balance:
            return "balance sheet"
        if has_cashflow:
            return "cash flow statement"
        if has_note:
            return "note table"
        return None

    @staticmethod
    def _parse_slot_struct(value: Any) -> Optional[dict[str, str]]:
        if not isinstance(value, dict):
            return None
        raw = value

        normalized: dict[str, str] = {}
        for key in ("entity", "period", "metric", "source_anchor", "unit", "rounding"):
            if key not in raw:
                continue
            val = raw.get(key)
            if val is None:
                continue
            if key == "source_anchor":
                anchor = SlotSupport._normalize_source_anchor(val)
                if anchor:
                    normalized[key] = anchor
                continue
            text = re.sub(r"\s+", " ", str(val).strip().lower())
            if not text or text in {"none", "null", "n/a"}:
                continue
            normalized[key] = text
        return normalized or None

    @staticmethod
    def _normalize_slot(value: Any) -> str:
        struct = SlotSupport._parse_slot_struct(value)
        if struct:
            ordered_keys = ("entity", "period", "metric", "source_anchor", "unit", "rounding")
            return "|".join(f"{key}={struct[key]}" for key in ordered_keys if key in struct)
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _normalize_slot_conflict_strategy(self, value: Any) -> str:
        strategy = str(value or "best_supported").strip().lower()
        if strategy in self._VALID_SLOT_CONFLICT_STRATEGIES:
            return strategy
        return "best_supported"

    def _ledger_value_conflict_key(self, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not text:
            return ""
        numeric_candidates = self._extract_scaled_numeric_candidates(text)
        if numeric_candidates:
            sig = max(numeric_candidates, key=lambda num: abs(num))
            return f"num:{sig:.12g}"
        return f"txt:{text}"

    def _required_slots(self, query_state: dict[str, Any]) -> list[Any]:
        slots_raw = query_state.get("required_slots", [])
        slots: list[Any] = []
        if isinstance(slots_raw, list):
            for item in slots_raw:
                struct = self._parse_slot_struct(item)
                if struct:
                    slots.append(struct)
                    continue
                text = str(item or "").strip()
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

    def _compute_missing_slots(
        self,
        query_state: dict[str, Any],
        evidence_ledger: list[dict[str, str]],
        slot_conflict_strategy: Optional[str] = None,
    ) -> list[Any]:
        required = self._required_slots(query_state)
        required_map = {self._normalize_slot(slot): slot for slot in required}
        values_by_slot: dict[str, set[str]] = {key: set() for key in required_map}
        conflict_strategy = self._normalize_slot_conflict_strategy(slot_conflict_strategy)

        for item in evidence_ledger:
            slot_key = self._normalize_slot(item.get("slot", ""))
            if slot_key not in required_map:
                continue
            value_norm = re.sub(r"\s+", " ", str(item.get("value", "") or "").strip().lower())
            if value_norm in {"missing", "n/a", "na", "none", "null", "unknown", "-"}:
                continue
            if value_norm:
                values_by_slot[slot_key].add(self._ledger_value_conflict_key(value_norm))

        missing: list[Any] = []
        for key, slot in required_map.items():
            values = values_by_slot.get(key, set())
            if len(values) == 0:
                missing.append(slot)
                continue
            if len(values) > 1 and conflict_strategy == "keep_missing_on_tie":
                missing.append(slot)
        return missing

    def _slot_period_group_key(self, slot_struct: dict[str, str]) -> str:
        entity = re.sub(r"\s+", " ", str(slot_struct.get("entity", "") or "").strip().lower())
        metric = self._canonical_metric_key(slot_struct.get("metric", ""))
        anchor = self._normalize_source_anchor(slot_struct.get("source_anchor")) or ""
        if not metric:
            return ""
        return f"{entity}|{metric}|{anchor}"

    def _collapsed_multi_period_slots(
        self,
        query_state: dict[str, Any],
        evidence_ledger: list[dict[str, str]],
    ) -> list[Any]:
        required = self._required_slots(query_state)
        if not required:
            return []
        required_map = {self._normalize_slot(slot): slot for slot in required}
        values_by_slot: dict[str, set[str]] = {key: set() for key in required_map}
        for item in evidence_ledger:
            slot_key = self._normalize_slot(item.get("slot", ""))
            if slot_key not in required_map:
                continue
            value_norm = re.sub(r"\s+", " ", str(item.get("value", "") or "").strip().lower())
            if not value_norm:
                continue
            values_by_slot[slot_key].add(self._ledger_value_conflict_key(value_norm))

        slots_by_group: dict[str, list[str]] = {}
        for slot_key, slot_raw in required_map.items():
            struct = self._parse_slot_struct(slot_raw)
            if not struct:
                continue
            years = set(self._extract_year_tokens(struct.get("period", "")))
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
                slot_struct = self._parse_slot_struct(required_map.get(slot_key, ""))
                period_years = tuple(
                    sorted(set(self._extract_year_tokens(str((slot_struct or {}).get("period", "")))))
                )
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

    def _sanitize_missing_slots(
        self,
        query_state: dict[str, Any],
        missing_slots_raw: Any,
    ) -> Optional[list[Any]]:
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

        if raw_nonempty and not seen:
            return None
        return [slot for slot in required if self._normalize_slot(slot) in seen]

    def _resolve_missing_slots(
        self,
        query_state: dict[str, Any],
        evidence_ledger: list[dict[str, str]],
        model_missing_slots: Optional[list[Any]],
        trust_model_missing: bool = False,
    ) -> list[Any]:
        required = self._required_slots(query_state)
        if not required:
            return []
        ledger_missing = self._compute_missing_slots(query_state, evidence_ledger)
        if not trust_model_missing:
            return ledger_missing
        if model_missing_slots is None:
            return ledger_missing
        if not evidence_ledger and not trust_model_missing:
            return required

        ledger_keys = {self._normalize_slot(slot) for slot in ledger_missing}
        model_keys = {self._normalize_slot(slot) for slot in model_missing_slots}
        target_keys = model_keys if trust_model_missing else (ledger_keys | model_keys)
        if not target_keys:
            return []
        return [slot for slot in required if self._normalize_slot(slot) in target_keys]

    def _slot_struct_matches(self, candidate: dict[str, str], required: dict[str, str]) -> bool:
        if not isinstance(candidate, dict) or not isinstance(required, dict):
            return False
        if not self._metric_matches(candidate.get("metric", ""), required.get("metric", "")):
            return False
        if not self._periods_overlap(candidate.get("period", ""), required.get("period", "")):
            return False
        if not self._entity_matches(candidate.get("entity", ""), required.get("entity", "")):
            return False
        required_anchor = str(required.get("source_anchor", "") or "").strip().lower()
        candidate_anchor = str(candidate.get("source_anchor", "") or "").strip().lower()
        if required_anchor and candidate_anchor and required_anchor != candidate_anchor:
            return False
        return True

    def _slot_key_by_year_hint(
        self,
        *,
        compatible_keys: list[str],
        required_map: dict[str, Any],
        citation: str,
        citation_span: str,
        value: str,
    ) -> Optional[str]:
        if len(compatible_keys) <= 1:
            return compatible_keys[0] if compatible_keys else None

        slot_years: dict[str, set[str]] = {}
        for key in compatible_keys:
            req_struct = self._parse_slot_struct(required_map.get(key, ""))
            if not req_struct:
                continue
            years = set(self._extract_year_tokens(str(req_struct.get("period", "") or "")))
            if years:
                slot_years[key] = years
        if not slot_years:
            return None

        title_years = set(self._extract_year_tokens(self._citation_doc_title(citation)))
        if title_years:
            title_matches = [
                key for key, years in slot_years.items() if not years.isdisjoint(title_years)
            ]
            if len(title_matches) == 1:
                return title_matches[0]

        value_nums = self._extract_scaled_numeric_candidates(str(value or ""))
        if not value_nums or not citation_span:
            return None
        span_text = str(citation_span or "")
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
                        tol = max(1e-6, abs(expected) * 1e-4)
                        if any(abs(found - expected) <= tol for found in window_nums):
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

    def _resolve_required_slot_key(
        self,
        slot_raw: Any,
        required_map: dict[str, Any],
        query_state: dict[str, Any],
        *,
        value: str = "",
        citation: str = "",
        citation_span: str = "",
    ) -> tuple[Optional[str], str]:
        direct_key = self._normalize_slot(slot_raw)
        if direct_key and direct_key in required_map:
            return direct_key, "exact_slot_match"

        slot_struct = self._parse_slot_struct(slot_raw)
        if not slot_struct:
            slot_text = str(slot_raw or "").strip()
            if not slot_text:
                return None, "missing_slot"
            canonical_slot = self._canonical_metric_key(slot_text)
            if not canonical_slot:
                return None, "slot_unmatched"
            matches: list[str] = []
            for req_key, req_slot in required_map.items():
                req_struct = self._parse_slot_struct(req_slot)
                req_metric = req_struct.get("metric", "") if req_struct else req_slot
                if self._metric_matches(canonical_slot, req_metric):
                    matches.append(req_key)
            if len(matches) == 1:
                return matches[0], "metric_fallback_match"
            if len(matches) > 1:
                return matches[0], "metric_fallback_ambiguous"
            return None, "slot_unmatched"

        query_entity = str(query_state.get("entity", "") or "").strip()
        compatible: list[str] = []
        mismatch_counts: dict[str, int] = {}
        for req_key, req_slot in required_map.items():
            req_struct = self._parse_slot_struct(req_slot)
            if not req_struct:
                continue
            if not self._metric_matches(slot_struct.get("metric", ""), req_struct.get("metric", "")):
                mismatch_counts["metric_mismatch"] = mismatch_counts.get("metric_mismatch", 0) + 1
                continue
            if not self._periods_overlap(slot_struct.get("period", ""), req_struct.get("period", "")):
                mismatch_counts["period_mismatch"] = mismatch_counts.get("period_mismatch", 0) + 1
                continue

            req_entity = str(req_struct.get("entity", "") or "").strip()
            cand_entity = str(slot_struct.get("entity", "") or "").strip()
            if self._is_generic_entity_label(req_entity) and not self._is_generic_entity_label(query_entity):
                req_entity = query_entity
            if self._is_generic_entity_label(cand_entity):
                cand_entity = ""
            if req_entity and cand_entity and not self._entity_matches(cand_entity, req_entity):
                mismatch_counts["entity_mismatch"] = mismatch_counts.get("entity_mismatch", 0) + 1
                continue

            req_anchor = str(req_struct.get("source_anchor", "") or "").strip().lower()
            cand_anchor = str(slot_struct.get("source_anchor", "") or "").strip().lower()
            if req_anchor and cand_anchor and req_anchor != cand_anchor:
                mismatch_counts["anchor_mismatch"] = mismatch_counts.get("anchor_mismatch", 0) + 1
                continue
            compatible.append(req_key)

        if len(compatible) == 1:
            return compatible[0], "structural_slot_match"
        if len(compatible) > 1:
            hinted = self._slot_key_by_year_hint(
                compatible_keys=compatible,
                required_map=required_map,
                citation=citation,
                citation_span=citation_span,
                value=value,
            )
            if hinted:
                return hinted, "structural_slot_tiebreak_year_hint"
            compatible.sort(key=lambda key: len(self._normalize_slot(required_map[key])), reverse=True)
            return compatible[0], "structural_slot_tiebreak"
        if mismatch_counts:
            reason = max(mismatch_counts.items(), key=lambda item: item[1])[0]
            return None, reason
        return None, "slot_unmatched"
