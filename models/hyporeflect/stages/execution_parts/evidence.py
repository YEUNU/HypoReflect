import logging
import re
from typing import Any, Optional

from models.hyporeflect.stages.common import CITATION_RE, normalize_missing_data_policy
from models.hyporeflect.stages.llm_json import generate_json_with_retries
from utils.prompts import (
    ENTRY_GATE_FORMAT_INSTRUCTION,
    ENTRY_GATE_PROMPT,
    ENTRY_GATE_RETRY_PROMPT,
    EVIDENCE_LEDGER_FORMAT_INSTRUCTION,
    EVIDENCE_LEDGER_PROMPT,
    EVIDENCE_LEDGER_RETRY_PROMPT,
    EVIDENCE_LEDGER_ZERO_RESCUE_PROMPT,
    MISSING_SLOT_RESCUE_PROMPT,
)


logger = logging.getLogger(__name__)


class EvidenceSupport:
    @staticmethod
    def _validate_ledger_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        entries = data.get("entries")
        if not isinstance(entries, list):
            return False, "entries must be JSON array"
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                return False, f"entries[{idx}] must be JSON object"
            if "slot" not in item:
                return False, f"entries[{idx}] missing slot"
            if not isinstance(item.get("slot"), (str, dict)):
                return False, f"entries[{idx}].slot must be string or object"
            if "value" not in item:
                return False, f"entries[{idx}] missing value"
            if not isinstance(item.get("value"), str):
                return False, f"entries[{idx}].value must be string"
            if "citation" not in item:
                return False, f"entries[{idx}] missing citation"
            if not isinstance(item.get("citation"), str):
                return False, f"entries[{idx}].citation must be string"
        missing_slots = data.get("missing_slots", [])
        if not isinstance(missing_slots, list):
            return False, "missing_slots must be JSON array"
        return True, ""

    def _ledger_retry_message(
        self,
        failed_output: Any,
        reason: str,
    ) -> str:
        return EVIDENCE_LEDGER_RETRY_PROMPT.format(
            error=reason,
            previous_output=self._compact_json(failed_output, max_chars=900),
        )

    @staticmethod
    def _validate_entry_gate_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        decisions = data.get("decisions")
        if not isinstance(decisions, list):
            return False, "decisions must be JSON array"
        for idx, item in enumerate(decisions):
            if not isinstance(item, dict):
                return False, f"decisions[{idx}] must be JSON object"
            if not isinstance(item.get("index"), int):
                return False, f"decisions[{idx}].index must be int"
            if not isinstance(item.get("keep"), bool):
                return False, f"decisions[{idx}].keep must be boolean"
            if not isinstance(item.get("reason", ""), str):
                return False, f"decisions[{idx}].reason must be string"
        return True, ""

    def _entry_gate_retry_message(self, failed_output: Any, reason: str) -> str:
        return ENTRY_GATE_RETRY_PROMPT.format(
            error=reason,
            previous_output=self._compact_json(failed_output, max_chars=900),
        )

    @staticmethod
    def _increment_reject_reason(reject_reasons: dict[str, int], reason: str) -> None:
        reject_reasons[reason] = int(reject_reasons.get(reason, 0) or 0) + 1

    @staticmethod
    def _init_ledger_diagnostics(entries_raw: Any) -> dict[str, Any]:
        return {
            "entries_raw": len(entries_raw) if isinstance(entries_raw, list) else 0,
            "accepted_entries": 0,
            "slot_match_fallback_count": 0,
            "rescue_invoked": False,
            "rescue_entries_raw": 0,
            "reject_reasons": {},
        }

    @staticmethod
    def _should_run_zero_entry_rescue(
        entries_raw: Any,
        required_slots: list[Any],
        context_excerpt: str,
    ) -> bool:
        return (
            isinstance(entries_raw, list)
            and len(entries_raw) == 0
            and bool(required_slots)
            and bool(str(context_excerpt or "").strip())
        )

    @staticmethod
    def _materialize_slot_entries(
        entries_by_slot: dict[str, dict[str, str]],
        conflict_slots: set[str],
        slot_conflict_strategy: str,
    ) -> list[dict[str, str]]:
        if slot_conflict_strategy == "keep_missing_on_tie" and conflict_slots:
            return [
                entry
                for key, entry in entries_by_slot.items()
                if key not in conflict_slots
            ]
        return list(entries_by_slot.values())

    @staticmethod
    def _needs_citation_repair(
        model_missing_slots: list[Any],
        context_excerpt: str,
        reject_reasons: dict[str, int],
    ) -> bool:
        if not model_missing_slots or not str(context_excerpt or "").strip():
            return False
        invalid_format = int(reject_reasons.get("invalid_citation_format", 0) or 0) > 0
        citation_not_in_context = int(reject_reasons.get("citation_not_in_context", 0) or 0) > 0
        return invalid_format or citation_not_in_context

    @staticmethod
    def _apply_entry_gate_diagnostics(
        diagnostics: dict[str, Any],
        gate_diag: dict[str, Any],
        fallback_kept_default: int,
    ) -> None:
        diagnostics["entry_gate_checked"] = int(gate_diag.get("checked", 0) or 0)
        diagnostics["entry_gate_rejected"] = int(gate_diag.get("rejected", 0) or 0)
        diagnostics["entry_gate_kept"] = int(gate_diag.get("kept", fallback_kept_default) or 0)
        diagnostics["entry_gate_fallback_applied"] = bool(gate_diag.get("fallback_applied", False))
        diagnostics["entry_gate_fallback_kept"] = int(gate_diag.get("fallback_kept", 0) or 0)
        diagnostics["entry_gate_coverage_regressed_slots"] = int(
            gate_diag.get("coverage_regressed_slots", 0) or 0
        )
        if gate_diag.get("bypassed"):
            diagnostics["entry_gate_bypassed"] = str(gate_diag.get("bypassed"))

    def _merge_entry_gate_reject_reasons(
        self,
        diagnostics: dict[str, Any],
        gate_diag: dict[str, Any],
    ) -> None:
        reject_reasons = diagnostics.get("reject_reasons", {})
        if not isinstance(reject_reasons, dict):
            return
        gate_reasons = gate_diag.get("reject_reasons", {})
        if not isinstance(gate_reasons, dict):
            return
        for reason, count in gate_reasons.items():
            reason_key = self._normalize_entry_gate_reason(reason)
            reject_key = f"entry_gate:{reason_key}"
            self._increment_reject_reason(reject_reasons, reject_key)
            if isinstance(count, int) and count > 1:
                reject_reasons[reject_key] += max(0, count - 1)

    @staticmethod
    def _normalize_entry_gate_reason(reason: Any) -> str:
        text = str(reason or "reject").strip()
        if not text:
            return "reject"
        lower = text.lower()
        if "forward-looking" in lower or "guidance" in lower or "expected" in lower or "planned" in lower:
            return "forward_looking_or_guidance"
        if "not directly ground" in lower:
            return "not_directly_grounded"
        if "narrative" in lower:
            return "narrative_or_non_statement"
        if "segment" in lower and "consolidated" in lower:
            return "segment_vs_consolidated_mismatch"
        if "period" in lower and "mismatch" in lower:
            return "period_mismatch"
        if "entity" in lower and "mismatch" in lower:
            return "entity_mismatch"
        return text[:120] if len(text) > 120 else text

    def _compute_slot_entry_score(
        self,
        *,
        slot_struct: Optional[dict[str, str]],
        citation_span: str,
        value: str,
    ) -> int:
        score = 0
        if citation_span:
            score += 1
            if self._value_grounded_in_span(value, citation_span):
                score += 2

        metric_hits = 0
        if slot_struct and citation_span:
            metric_terms = [
                term
                for term in self._metric_alias_terms(slot_struct.get("metric", ""))
                if len(str(term).strip()) >= 4
            ][:8]
            lower_span = citation_span.lower()
            metric_hits = sum(1 for term in metric_terms if str(term).lower() in lower_span)

        score += min(2, metric_hits)
        return score

    def _apply_slot_entry_candidate(
        self,
        *,
        slot_key: str,
        candidate_entry: dict[str, str],
        value_key: str,
        citation: str,
        candidate_score: int,
        slot_conflict_strategy: str,
        entries_by_slot: dict[str, dict[str, str]],
        entry_score_by_slot: dict[str, int],
        entry_value_key_by_slot: dict[str, str],
        slot_seen_keys_by_slot: dict[str, set[tuple[str, str]]],
        conflict_slots: set,
    ) -> Optional[str]:
        seen_keys = slot_seen_keys_by_slot.setdefault(slot_key, set())
        dedupe_key = (
            value_key,
            self._normalize_citation(citation),
        )
        if dedupe_key in seen_keys:
            return "duplicate_slot_candidate"
        seen_keys.add(dedupe_key)

        existing = entries_by_slot.get(slot_key)
        existing_score = int(entry_score_by_slot.get(slot_key, -1) or -1)
        existing_value_key = entry_value_key_by_slot.get(slot_key, "")
        if existing is None:
            entries_by_slot[slot_key] = candidate_entry
            entry_score_by_slot[slot_key] = candidate_score
            entry_value_key_by_slot[slot_key] = value_key
            return None

        if value_key and existing_value_key and value_key == existing_value_key:
            if candidate_score > existing_score:
                entries_by_slot[slot_key] = candidate_entry
                entry_score_by_slot[slot_key] = candidate_score
                entry_value_key_by_slot[slot_key] = value_key
            return "duplicate_slot_same_value"

        if slot_conflict_strategy == "keep_missing_on_tie":
            conflict_slots.add(slot_key)
            return "slot_conflict_keep_missing"

        if candidate_score > existing_score:
            entries_by_slot[slot_key] = candidate_entry
            entry_score_by_slot[slot_key] = candidate_score
            entry_value_key_by_slot[slot_key] = value_key
        return "slot_conflict_best_supported"

    async def _llm_gate_entries(
        self,
        query_state: dict[str, Any],
        context_excerpt: str,
        entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not entries:
            return entries, {
                "checked": 0,
                "kept": 0,
                "rejected": 0,
                "reject_reasons": {},
                "fallback_applied": False,
                "fallback_kept": 0,
                "coverage_regressed_slots": 0,
            }

        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        if answer_type == "compute":
            return entries, {
                "checked": 0,
                "kept": len(entries),
                "rejected": 0,
                "reject_reasons": {},
                "fallback_applied": False,
                "fallback_kept": 0,
                "coverage_regressed_slots": 0,
                "bypassed": "compute_priority",
            }

        citation_map = self._context_citation_map(context_excerpt)
        candidates: list[dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            citation = str(entry.get("citation", "") or "").strip()
            citation_key = self._normalize_citation(citation)
            candidates.append(
                {
                    "index": idx,
                    "slot": entry.get("slot"),
                    "value": str(entry.get("value", "") or "").strip(),
                    "citation": citation,
                    "span": citation_map.get(citation_key, ""),
                }
            )

        prompt = ENTRY_GATE_PROMPT.format(
            query_state=self._compact_json(query_state, max_chars=1200),
            candidates=self._compact_json(candidates, max_chars=5000),
            context=context_excerpt,
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": ENTRY_GATE_FORMAT_INSTRUCTION},
        ]
        data, ok, _ = await generate_json_with_retries(
            self.llm,
            messages,
            self._validate_entry_gate_json,
            self._entry_gate_retry_message,
            max_attempts=2,
            logger=logger,
            warning_prefix="entry gate json generation failed",
            model=self.stage_model,
        )
        if not ok or not isinstance(data, dict):
            return entries, {"checked": 0, "kept": len(entries), "rejected": 0, "reject_reasons": {}}

        decisions_raw = data.get("decisions", [])
        decision_map: dict[int, dict[str, Any]] = {}
        if isinstance(decisions_raw, list):
            for item in decisions_raw:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if not isinstance(idx, int):
                    continue
                decision_map[idx] = {
                    "keep": bool(item.get("keep", False)),
                    "reason": str(item.get("reason", "") or "").strip(),
                }

        kept: list[dict[str, Any]] = []
        reject_reasons: dict[str, int] = {}
        for idx, entry in enumerate(entries):
            decision = decision_map.get(idx)
            if decision is None:
                kept.append(entry)
                continue
            if decision["keep"]:
                kept.append(entry)
                continue
            reason_key = decision["reason"] or "entry_gate_reject"
            reject_reasons[reason_key] = reject_reasons.get(reason_key, 0) + 1

        baseline_missing = {
            self._normalize_slot(slot)
            for slot in self._compute_missing_slots(query_state, entries)
        }
        gated_missing = {
            self._normalize_slot(slot)
            for slot in self._compute_missing_slots(query_state, kept)
        }
        coverage_regressed = {
            key for key in gated_missing if key and key not in baseline_missing
        }

        fallback_applied = False
        fallback_kept = 0
        kept_slot_keys = {
            self._normalize_slot(entry.get("slot", ""))
            for entry in kept
            if self._normalize_slot(entry.get("slot", ""))
        }

        if coverage_regressed:
            fallback_applied = True
            for entry in entries:
                slot_key = self._normalize_slot(entry.get("slot", ""))
                if not slot_key or slot_key in kept_slot_keys:
                    continue
                if slot_key not in coverage_regressed:
                    continue
                kept.append(entry)
                kept_slot_keys.add(slot_key)
                fallback_kept += 1

        if not kept and entries:
            fallback_applied = True
            for entry in entries:
                slot_key = self._normalize_slot(entry.get("slot", ""))
                if slot_key and slot_key in kept_slot_keys:
                    continue
                kept.append(entry)
                if slot_key:
                    kept_slot_keys.add(slot_key)
                fallback_kept += 1
                if fallback_kept >= 1:
                    break

        return kept, {
            "checked": len(entries),
            "kept": len(kept),
            "rejected": max(0, len(entries) - len(kept)),
            "reject_reasons": reject_reasons,
            "fallback_applied": fallback_applied,
            "fallback_kept": fallback_kept,
            "coverage_regressed_slots": len(coverage_regressed),
        }

    async def _rescue_missing_slot_entries(
        self,
        query_state: dict[str, Any],
        context_excerpt: str,
        missing_slots: list[Any],
    ) -> list[dict[str, Any]]:
        if not missing_slots or not context_excerpt.strip():
            return []
        allowed_citations: list[str] = []
        seen = set()
        for match in CITATION_RE.finditer(context_excerpt):
            citation = str(match.group(0) or "").strip()
            key = self._normalize_citation(citation)
            if not key or key in seen:
                continue
            seen.add(key)
            allowed_citations.append(citation)
        if not allowed_citations:
            return []

        prompt = MISSING_SLOT_RESCUE_PROMPT.format(
            query_state=self._compact_json(query_state, max_chars=1200),
            missing_slots=self._compact_json(missing_slots, max_chars=2000),
            allowed_citations=self._compact_json(allowed_citations, max_chars=2400),
            context=context_excerpt,
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": EVIDENCE_LEDGER_FORMAT_INSTRUCTION},
        ]
        data, ok, _ = await generate_json_with_retries(
            self.llm,
            messages,
            self._validate_ledger_json,
            self._ledger_retry_message,
            max_attempts=3,
            logger=logger,
            warning_prefix="missing-slot rescue json generation failed",
            model=self.stage_model,
        )
        if not ok or not isinstance(data, dict):
            data = {}
        entries_raw = data.get("entries", [])
        if isinstance(entries_raw, list) and entries_raw:
            return entries_raw

        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        if answer_type == "compute":
            context_nodes = self._context_excerpt_nodes(context_excerpt)
            return self._deterministic_compute_slot_entries(
                query_state=query_state,
                missing_slots=missing_slots,
                nodes=context_nodes,
            )
        return entries_raw if isinstance(entries_raw, list) else []

    @staticmethod
    def _node_citation(node: dict[str, Any]) -> str:
        title = str(node.get("title") or node.get("doc") or "Unknown")
        page = node.get("page", 0)
        chunk_id = node.get("sent_id", -1)
        return f"[[{title}, Page {page}, Chunk {chunk_id}]]"

    def _dedupe_nodes(self, nodes: list[dict[str, Any]], max_nodes: Optional[int] = None) -> list[dict[str, Any]]:
        cap = max_nodes if isinstance(max_nodes, int) and max_nodes > 0 else self.context_node_budget
        seen = set()
        deduped: list[dict[str, Any]] = []
        for node in reversed(nodes):
            if not isinstance(node, dict):
                continue
            citation = self._node_citation(node)
            if citation in seen:
                continue
            seen.add(citation)
            deduped.append(node)
            if len(deduped) >= cap:
                break
        deduped.reverse()
        return deduped

    def _serialize_nodes(
        self,
        nodes: list[dict[str, Any]],
        query_state: Optional[dict[str, Any]] = None,
        max_text_chars: int = 760,
    ) -> str:
        if (
            isinstance(query_state, dict)
            and str(query_state.get("answer_type", "")).lower() == "compute"
            and max_text_chars < 1200
        ):
            max_text_chars = 1200
        chunks: list[str] = []
        for node in nodes:
            text = self._extract_relevant_span(
                str(node.get("text", "") or ""),
                query_state=query_state,
                max_chars=max_text_chars,
            )
            if not text:
                continue
            chunks.append(f"{self._node_citation(node)}\n{text}")
        return "\n\n".join(chunks)

    def _is_insufficient_answer(self, answer: str) -> bool:
        answer_lower = str(answer or "").lower()
        return any(marker in answer_lower for marker in self._INSUFFICIENT_ANSWER_MARKERS)

    def _extract_answer_citations(self, answer: str) -> list[str]:
        return [m.group(0).strip() for m in CITATION_RE.finditer(str(answer or ""))]

    def _is_zero_policy_answer(self, answer: str) -> bool:
        text = str(answer or "").strip()
        if not text:
            return False
        without_citations = CITATION_RE.sub("", text)
        normalized = re.sub(r"\s+", " ", without_citations).strip().lower()
        return normalized in self._ZERO_POLICY_ANSWERS

    def _verify_answer_grounding(
        self,
        answer: str,
        query_state: dict[str, Any],
        evidence_ledger: list[dict[str, str]],
        context: str,
        missing_slots: list[Any],
    ) -> tuple[bool, str]:
        text = str(answer or "").strip()
        missing_data_policy = normalize_missing_data_policy(
            query_state.get("missing_data_policy")
        )
        if not text:
            return False, "empty final answer"
        if "@@ANSWER:" not in text:
            return False, "missing @@ANSWER prefix"
        if len(re.findall(r"@@ANSWER:", text, flags=re.IGNORECASE)) > 1:
            return False, "multiple @@ANSWER prefixes"

        if self._is_insufficient_answer(text):
            if missing_slots:
                if missing_data_policy == "zero_if_not_explicit":
                    return False, "query policy requires zero when value is not explicitly outlined"
                if missing_data_policy == "inapplicable_explain":
                    return False, "query policy requires inapplicable explanation, not insufficient evidence"
                return True, ""
            return False, "insufficient evidence despite no missing slots"

        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        if answer_type == "boolean":
            if self._extract_boolean_label(text) is None:
                return False, "boolean answer must start with yes/no"

        if (
            missing_data_policy == "zero_if_not_explicit"
            and missing_slots
            and self._is_zero_policy_answer(text)
        ):
            return True, ""

        citations = self._extract_answer_citations(text)
        if not citations:
            return False, "missing inline citations"

        context_citations = self._context_citation_map(context)
        ledger_citation_keys = {
            self._normalize_citation(entry.get("citation", ""))
            for entry in evidence_ledger
            if self._normalize_citation(entry.get("citation", ""))
        }
        is_compute = str(query_state.get("answer_type", "")).lower() == "compute"
        if context_citations:
            for citation in citations:
                key = self._normalize_citation(citation)
                if key not in context_citations:
                    if is_compute and key in ledger_citation_keys:
                        continue
                    return False, f"citation not present in context: {citation}"

        if is_compute:
            if missing_slots:
                return False, "compute answer requires all slots grounded"
            answer_citation_keys = {
                self._normalize_citation(citation)
                for citation in citations
            }
            slot_to_citations: dict[str, set[str]] = {}
            for entry in evidence_ledger:
                slot_key = self._normalize_slot(entry.get("slot", ""))
                citation_key = self._normalize_citation(entry.get("citation", ""))
                if not slot_key or not citation_key:
                    continue
                slot_to_citations.setdefault(slot_key, set()).add(citation_key)
            for slot in self._required_slots(query_state):
                slot_key = self._normalize_slot(slot)
                slot_citations = slot_to_citations.get(slot_key, set())
                if slot_citations and answer_citation_keys.isdisjoint(slot_citations):
                    return False, f"missing citation coverage for slot: {slot_key}"

        return True, ""

    async def _extract_evidence_entries(
        self,
        query_state: dict[str, Any],
        context_excerpt: str,
        filter_policy: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, str]], Optional[list[Any]], dict[str, Any]]:
        if not context_excerpt.strip():
            return [], None, {"entries_raw": 0, "accepted_entries": 0, "reject_reasons": {"empty_context": 1}}
        required_slots = self._required_slots(query_state)
        required_map = {self._normalize_slot(slot): slot for slot in required_slots}
        base_messages = [
            {
                "role": "user",
                "content": EVIDENCE_LEDGER_PROMPT.format(
                    query_state=self._compact_json(query_state, max_chars=1000),
                    filter_policy=self._compact_json(filter_policy or {}, max_chars=1200),
                    context=context_excerpt,
                ),
            },
            {"role": "user", "content": EVIDENCE_LEDGER_FORMAT_INSTRUCTION},
        ]
        data, _, _ = await generate_json_with_retries(
            self.llm,
            base_messages,
            self._validate_ledger_json,
            self._ledger_retry_message,
            max_attempts=3,
            logger=logger,
            warning_prefix="evidence ledger json generation failed",
            model=self.stage_model,
        )
        entries_raw = data.get("entries", [])
        diagnostics: dict[str, Any] = self._init_ledger_diagnostics(entries_raw)

        if self._should_run_zero_entry_rescue(entries_raw, required_slots, context_excerpt):
            diagnostics["rescue_invoked"] = True
            rescue_messages = [
                {
                    "role": "user",
                    "content": EVIDENCE_LEDGER_ZERO_RESCUE_PROMPT.format(
                        query_state=self._compact_json(query_state, max_chars=1000),
                        filter_policy=self._compact_json(filter_policy or {}, max_chars=1200),
                        context=context_excerpt,
                    ),
                },
                {"role": "user", "content": EVIDENCE_LEDGER_FORMAT_INSTRUCTION},
            ]
            rescue_data, _, _ = await generate_json_with_retries(
                self.llm,
                rescue_messages,
                self._validate_ledger_json,
                self._ledger_retry_message,
                max_attempts=2,
                logger=logger,
                warning_prefix="evidence ledger rescue json generation failed",
                model=self.stage_model,
            )
            rescue_entries = rescue_data.get("entries", [])
            if isinstance(rescue_entries, list):
                entries_raw = rescue_entries
                diagnostics["rescue_entries_raw"] = len(rescue_entries)
            if isinstance(rescue_data, dict) and "missing_slots" in rescue_data:
                data = rescue_data

        reject_reasons = diagnostics["reject_reasons"]

        def reject(reason: str) -> None:
            self._increment_reject_reason(reject_reasons, reason)

        if not isinstance(entries_raw, list):
            reject("entries_not_list")
            return [], self._compute_missing_slots(query_state, []), diagnostics

        context_citations = self._context_citation_map(context_excerpt)
        placeholder_values = {"missing", "n/a", "na", "none", "null", "unknown", "-"}
        entries_by_slot: dict[str, dict[str, str]] = {}
        entry_score_by_slot: dict[str, int] = {}
        entry_value_key_by_slot: dict[str, str] = {}
        slot_seen_keys_by_slot: dict[str, set[tuple[str, str]]] = {}
        conflict_slots = set()
        query_entity = str(query_state.get("entity", "") or "").strip()
        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        slot_conflict_strategy = self._normalize_slot_conflict_strategy(
            (filter_policy or {}).get("slot_conflict_strategy", "best_supported")
        )
        diagnostics["slot_conflict_strategy"] = slot_conflict_strategy

        def materialize_entries() -> list[dict[str, str]]:
            return self._materialize_slot_entries(
                entries_by_slot=entries_by_slot,
                conflict_slots=conflict_slots,
                slot_conflict_strategy=slot_conflict_strategy,
            )

        def consume_entries(items: list[Any]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    reject("entry_not_object")
                    continue
                slot_raw = item.get("slot", "")
                value = str(item.get("value", "") or "").strip()
                citation = self._coerce_citation(item.get("citation", ""))
                if not slot_raw:
                    reject("missing_slot")
                    continue
                if not value:
                    reject("missing_value")
                    continue
                if not citation:
                    reject("missing_citation")
                    continue
                if CITATION_RE.search(citation) is None:
                    reject("invalid_citation_format")
                    continue
                citation_key = self._normalize_citation(citation)
                if context_citations and citation_key not in context_citations:
                    reject("citation_not_in_context")
                    continue
                citation_span = context_citations.get(citation_key, "")
                slot_key, slot_match_reason = self._resolve_required_slot_key(
                    slot_raw=slot_raw,
                    required_map=required_map,
                    query_state=query_state,
                    value=value,
                    citation=citation,
                    citation_span=citation_span,
                )
                if slot_key is None:
                    reject(slot_match_reason)
                    continue
                if slot_match_reason != "exact_slot_match":
                    diagnostics["slot_match_fallback_count"] += 1
                value_grounded = True
                if citation_span and not self._value_grounded_in_span(value, citation_span):
                    value_grounded = False
                slot_struct = self._parse_slot_struct(required_map[slot_key])
                if slot_struct:
                    slot_entity = str(slot_struct.get("entity", "") or "").strip()
                    if self._is_generic_entity_label(slot_entity) and not self._is_generic_entity_label(query_entity):
                        slot_entity = query_entity
                    citation_title = self._citation_doc_title(citation)
                    citation_prefix = citation_title.split("_", 1)[0].strip() if citation_title else ""
                    if slot_entity and citation_prefix and not self._entity_matches(slot_entity, citation_prefix):
                        reject("entity_mismatch")
                        continue
                    slot_period = str(slot_struct.get("period", "") or "").strip()
                    if not slot_period:
                        slot_period = str(query_state.get("period", "") or "").strip()
                    slot_years = set(self._extract_year_tokens(slot_period))
                    citation_years = set(self._extract_year_tokens(citation_title))
                    if (
                        answer_type != "compute"
                        and slot_years
                        and citation_years
                        and slot_years.isdisjoint(citation_years)
                    ):
                        reject("citation_period_mismatch")
                        continue
                    if value_grounded and not self._value_matches_slot_period(
                        value=value,
                        slot_period=slot_period,
                        citation=citation,
                        citation_span=citation_span,
                    ):
                        reject("value_period_mismatch")
                        continue
                    slot_anchor = str(slot_struct.get("source_anchor", "") or "").strip().lower()
                    query_anchor = str(query_state.get("source_anchor", "") or "").strip().lower()
                    anchor_to_check = slot_anchor or query_anchor
                    if not value_grounded:
                        relaxed_capex_match = (
                            self._is_capex_metric(slot_struct.get("metric", ""))
                            and anchor_to_check == "cash flow statement"
                            and self._contains_numeric_token(citation_span)
                            and any(
                                marker in citation_span.lower()
                                for marker in self._CAPEX_RELAXED_GROUNDING_MARKERS
                            )
                        )
                        if not relaxed_capex_match:
                            reject("value_not_in_cited_span")
                            continue
                    lower_span = citation_span.lower() if citation_span else ""
                    metric_key = self._canonical_metric_key(slot_struct.get("metric", ""))
                    metric_terms = [
                        term
                        for term in self._metric_alias_terms(slot_struct.get("metric", ""))
                        if len(str(term).strip()) >= 4
                    ][:12]
                    if (
                        answer_type == "compute"
                        and citation_span
                        and metric_terms
                        and not self._is_capex_metric(slot_struct.get("metric", ""))
                        and not self._value_near_metric_term(
                            value=value,
                            citation_span=citation_span,
                            metric_terms=metric_terms,
                        )
                    ):
                        reject("value_not_near_metric_term")
                        continue
                    expects_ratio_value = self._metric_expects_ratio_value(metric_key)
                    value_lower = str(value or "").lower()
                    if not expects_ratio_value and ("%" in value_lower or "percent" in value_lower):
                        reject("percent_value_for_amount_slot")
                        continue

                    if anchor_to_check == "cash flow statement":
                        if self._is_dividend_metric(slot_struct.get("metric", "")) and "dividend" in citation_span.lower():
                            pass
                        else:
                            if not self._is_anchor_grounded_in_citation(
                                source_anchor=anchor_to_check,
                                citation=citation,
                                citation_span=citation_span,
                            ):
                                reject("source_anchor_mismatch")
                                continue
                    if self._is_capex_metric(slot_struct.get("metric", "")) and anchor_to_check == "cash flow statement":
                        if not any(marker in lower_span for marker in self._CAPEX_AMOUNT_MARKERS):
                            reject("capex_marker_missing")
                            continue
                        if any(marker in lower_span for marker in self._CAPEX_RATIO_SPAN_MARKERS):
                            reject("capex_ratio_span")
                            continue
                        value = self._positive_amount_string(value)
                value_norm = re.sub(r"\s+", " ", value.strip().lower())
                if value_norm in placeholder_values:
                    reject("placeholder_value")
                    continue
                candidate_entry = {
                    "slot": required_map[slot_key],
                    "value": value,
                    "citation": citation,
                }
                value_key = self._ledger_value_conflict_key(value)
                candidate_score = self._compute_slot_entry_score(
                    slot_struct=slot_struct,
                    citation_span=citation_span,
                    value=value,
                )
                reject_reason = self._apply_slot_entry_candidate(
                    slot_key=slot_key,
                    candidate_entry=candidate_entry,
                    value_key=value_key,
                    citation=citation,
                    candidate_score=candidate_score,
                    slot_conflict_strategy=slot_conflict_strategy,
                    entries_by_slot=entries_by_slot,
                    entry_score_by_slot=entry_score_by_slot,
                    entry_value_key_by_slot=entry_value_key_by_slot,
                    slot_seen_keys_by_slot=slot_seen_keys_by_slot,
                    conflict_slots=conflict_slots,
                )
                if reject_reason:
                    reject(reject_reason)

        consume_entries(entries_raw)
        entries = materialize_entries()
        diagnostics["accepted_entries"] = len(entries)
        model_missing_slots = self._compute_missing_slots(
            query_state,
            entries,
            slot_conflict_strategy=slot_conflict_strategy,
        )
        context_citation_count = len(context_citations)
        needs_citation_repair = self._needs_citation_repair(
            model_missing_slots=model_missing_slots,
            context_excerpt=context_excerpt,
            reject_reasons=diagnostics["reject_reasons"],
        )
        needs_slot_rescue = self._should_force_missing_slot_rescue(
            query_state=query_state,
            model_missing_slots=model_missing_slots,
            diagnostics=diagnostics,
            context_citation_count=context_citation_count,
        )
        if needs_citation_repair or needs_slot_rescue:
            diagnostics["rescue_invoked"] = True
            if needs_citation_repair:
                diagnostics["citation_repair_invoked"] = True
            if needs_slot_rescue:
                diagnostics["slot_rescue_invoked"] = True
            repair_entries_raw = await self._rescue_missing_slot_entries(
                query_state=query_state,
                context_excerpt=context_excerpt,
                missing_slots=model_missing_slots,
            )
            diagnostics["rescue_entries_raw"] = max(
                int(diagnostics.get("rescue_entries_raw", 0) or 0),
                len(repair_entries_raw),
            )
            if repair_entries_raw:
                consume_entries(repair_entries_raw)
                entries = materialize_entries()
                diagnostics["accepted_entries"] = len(entries)

        if entries:
            gated_entries, gate_diag = await self._llm_gate_entries(
                query_state=query_state,
                context_excerpt=context_excerpt,
                entries=entries,
            )
            self._apply_entry_gate_diagnostics(
                diagnostics=diagnostics,
                gate_diag=gate_diag,
                fallback_kept_default=len(entries),
            )
            self._merge_entry_gate_reject_reasons(
                diagnostics=diagnostics,
                gate_diag=gate_diag,
            )
            entries = gated_entries
            diagnostics["accepted_entries"] = len(entries)

        model_missing_slots = self._compute_missing_slots(
            query_state,
            entries,
            slot_conflict_strategy=slot_conflict_strategy,
        )
        return entries, model_missing_slots, diagnostics

    def _should_force_missing_slot_rescue(
        self,
        *,
        query_state: dict[str, Any],
        model_missing_slots: list[Any],
        diagnostics: dict[str, Any],
        context_citation_count: int,
    ) -> bool:
        if not isinstance(model_missing_slots, list) or not model_missing_slots:
            return False

        accepted_entries = int(diagnostics.get("accepted_entries", 0) or 0)
        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        if answer_type == "compute":
            if context_citation_count < 1:
                return False
            return True
        if accepted_entries > 0:
            return False

        min_citations = 4
        if context_citation_count < min_citations:
            reject_reasons = diagnostics.get("reject_reasons", {})
            if isinstance(reject_reasons, dict) and int(reject_reasons.get("value_period_mismatch", 0) or 0) > 0:
                return context_citation_count >= 1
            return False

        reject_reasons = diagnostics.get("reject_reasons", {})
        if isinstance(reject_reasons, dict) and int(reject_reasons.get("value_period_mismatch", 0) or 0) > 0:
            return True
        reject_total = 0
        if isinstance(reject_reasons, dict):
            for reason, count in reject_reasons.items():
                if str(reason).startswith("entry_gate:"):
                    continue
                try:
                    reject_total += int(count)
                except Exception:
                    reject_total += 1
        if reject_total >= 2:
            return True

        required_count = len(self._required_slots(query_state))
        return required_count >= 2
