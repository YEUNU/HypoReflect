import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from core.config import RAGConfig
from models.hyporeflect.state import AgentState
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


@dataclass
class ToolSearchPlan:
    entities: list[str]
    search_entities: list[str]
    depth: int
    top_k: int


class SearchSupport:
    def _build_search_entities(
        self,
        query: str,
        query_state: dict[str, Any],
        missing_slots: list[Any],
    ) -> list[str]:
        candidates: list[str] = []
        answer_type = str(query_state.get("answer_type", "")).strip().lower()
        query_entity = str(query_state.get("entity", "") or "").strip()
        if answer_type == "compute" and isinstance(missing_slots, list):
            # Prioritize per-slot structured intents first so multi-year compute
            # queries keep cross-year coverage under candidate caps.
            for slot in missing_slots:
                slot_struct = self._parse_slot_struct(slot)
                if not slot_struct:
                    continue
                slot_entity = str(slot_struct.get("entity", "") or "").strip()
                if self._is_generic_entity_label(slot_entity):
                    slot_entity = ""
                if not slot_entity and query_entity and not self._is_generic_entity_label(query_entity):
                    slot_entity = query_entity
                slot_period = str(slot_struct.get("period", "") or "").strip()
                slot_metric = self._normalize_metric_text(slot_struct.get("metric", ""))
                slot_anchor = str(slot_struct.get("source_anchor", "") or "").strip()
                if slot_entity and slot_period and slot_metric and slot_anchor:
                    candidates.append(f"{slot_entity} {slot_period} {slot_metric} {slot_anchor}")
                if slot_entity and slot_period and slot_metric:
                    candidates.append(f"{slot_entity} {slot_period} {slot_metric}")
                if slot_entity and slot_metric and slot_anchor:
                    candidates.append(f"{slot_entity} {slot_metric} {slot_anchor}")
                if slot_period and slot_metric and slot_anchor:
                    candidates.append(f"{slot_period} {slot_metric} {slot_anchor}")
                if slot_period and slot_metric:
                    candidates.append(f"{slot_period} {slot_metric}")
                if slot_metric and slot_anchor:
                    candidates.append(f"{slot_metric} {slot_anchor}")
                if slot_period:
                    candidates.append(slot_period)

        for field in ("entity", "period", "metric", "source_anchor"):
            value = str(query_state.get(field, "") or "").strip()
            if value:
                candidates.append(value)
        for alias in self._entity_search_aliases(query_state.get("entity", "")):
            candidates.append(alias)
        top_metric = self._normalize_metric_text(query_state.get("metric", ""))
        if top_metric:
            top_alias_cap = 10 if answer_type == "list" else 6
            for metric_alias in self._metric_alias_terms(top_metric)[:top_alias_cap]:
                candidates.append(metric_alias)

        for slot in missing_slots:
            slot_struct = self._parse_slot_struct(slot)
            if slot_struct:
                slot_entity = str(slot_struct.get("entity", "") or "").strip()
                slot_period = str(slot_struct.get("period", "") or "").strip()
                slot_metric = self._normalize_metric_text(slot_struct.get("metric", ""))
                slot_anchor = str(slot_struct.get("source_anchor", "") or "").strip()
                if slot_entity:
                    candidates.append(slot_entity)
                    for alias in self._entity_search_aliases(slot_entity):
                        candidates.append(alias)
                if slot_period:
                    candidates.append(slot_period)
                if slot_metric:
                    candidates.append(slot_metric)
                    metric_key = self._canonical_metric_key(slot_metric)
                    slot_alias_cap = 6
                    if answer_type == "list":
                        slot_alias_cap = 10
                    if "debt securities" in metric_key or "registered to trade" in metric_key:
                        slot_alias_cap = 14
                    for metric_alias in self._metric_alias_terms(slot_metric)[:slot_alias_cap]:
                        candidates.append(metric_alias)
                if slot_anchor:
                    candidates.append(slot_anchor)
                    for anchor_kw in self._source_anchor_keywords(slot_anchor):
                        candidates.append(anchor_kw)
                    for marker in self._source_anchor_strict_markers(slot_anchor)[:4]:
                        candidates.append(marker)
                if slot_entity and slot_period and slot_metric:
                    candidates.append(f"{slot_entity} {slot_period} {slot_metric}")
                if slot_entity and slot_period and slot_metric and slot_anchor:
                    candidates.append(f"{slot_entity} {slot_period} {slot_metric} {slot_anchor}")
            else:
                slot_text = str(slot).strip()
                if slot_text:
                    candidates.append(slot_text)

        entity = str(query_state.get("entity", "") or "").strip()
        period = str(query_state.get("period", "") or "").strip()
        metric = str(query_state.get("metric", "") or "").strip()
        source_anchor = str(query_state.get("source_anchor", "") or "").strip()
        if entity and period and metric:
            candidates.append(f"{entity} {period} {metric}")
        if entity and period and metric and source_anchor:
            candidates.append(f"{entity} {period} {metric} {source_anchor}")
        if source_anchor:
            for anchor_kw in self._source_anchor_keywords(source_anchor):
                candidates.append(anchor_kw)
            for marker in self._source_anchor_strict_markers(source_anchor)[:4]:
                candidates.append(marker)

        if query.strip():
            candidates.append(query.strip())

        deduped: list[str] = []
        seen = set()
        for value in candidates:
            key = value.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        required_count = len(self._required_slots(query_state))
        missing_count = len(missing_slots) if isinstance(missing_slots, list) else 0
        if answer_type == "compute":
            if missing_count >= 4:
                cap = 30
            elif missing_count >= 2:
                cap = 24
            else:
                cap = 20
        elif required_count >= 2 or missing_count >= 2:
            cap = 24
        else:
            cap = 12
        return deduped[:cap]

    async def _call_graph_search(
        self,
        entities: list[str],
        depth: int,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            result = self.grag.graph_search(entities, depth=depth, top_k=top_k)
            if asyncio.iscoroutine(result):
                result = await result
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], list)
            ):
                return str(result[0] or ""), result[1]
        except Exception as exc:
            logger.warning("graph_search call failed: %s", exc)
        return "", []

    @staticmethod
    def _unpack_evidence_entries_result(
        result: Any,
    ) -> tuple[list[dict[str, str]], Optional[list[Any]], dict[str, Any]]:
        """
        Backward-compatible unpacking:
        legacy tests/mocks may return (entries, missing_slots) while
        current implementation returns (entries, missing_slots, diagnostics).
        """
        if isinstance(result, tuple):
            if len(result) == 3:
                entries, missing_slots, diagnostics = result
            elif len(result) == 2:
                entries, missing_slots = result
                diagnostics = {}
            else:
                return [], None, {"invalid_result_arity": len(result)}
        else:
            return [], None, {"invalid_result_type": type(result).__name__}

        entries_list = entries if isinstance(entries, list) else []
        missing_list = missing_slots if isinstance(missing_slots, list) else None
        diagnostics_obj = diagnostics if isinstance(diagnostics, dict) else {}
        return entries_list, missing_list, diagnostics_obj

    def _merge_ledger(
        self,
        current: list[dict[str, str]],
        new_entries: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not new_entries:
            return list(current)

        new_slot_keys = {
            self._normalize_slot(entry.get("slot", ""))
            for entry in new_entries
            if self._normalize_slot(entry.get("slot", ""))
        }
        merged = [
            entry
            for entry in current
            if self._normalize_slot(entry.get("slot", "")) not in new_slot_keys
        ]

        seen = set()
        for entry in new_entries:
            key = (
                self._normalize_slot(entry.get("slot", "")),
                str(entry.get("value", "") or ""),
                str(entry.get("citation", "") or ""),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            merged.append(entry)
        return merged

    def _resolve_tool_top_k(self, state: AgentState, args: dict[str, Any]) -> int:
        top_k = args.get("top_k", RAGConfig.DEFAULT_TOP_K)
        try:
            top_k = max(1, min(int(top_k), 8))
        except Exception:
            top_k = min(max(1, RAGConfig.DEFAULT_TOP_K), 8)

        answer_type = str(state.query_state.get("answer_type", "")).lower()
        missing_count = len(state.missing_slots) if isinstance(state.missing_slots, list) else 0
        if missing_count:
            if answer_type == "compute":
                top_k = max(top_k, 7)
            else:
                top_k = max(top_k, 6)
            if missing_count >= 2:
                top_k = max(top_k, 8)
        return top_k

    @staticmethod
    def _turn_entity_cap(answer_type: str, required_count: int, missing_count: int) -> int:
        if answer_type == "compute":
            if required_count >= 4 or missing_count >= 4:
                return 28
            return 22
        if required_count >= 2 or missing_count >= 2:
            return 18
        return 12

    def _build_tool_search_plan(
        self,
        state: AgentState,
        routed_tool: str,
        routed_args: dict[str, Any],
        loop_state: Any,
        top_k: int,
    ) -> ToolSearchPlan:
        answer_type = str(state.query_state.get("answer_type", "")).lower()
        missing_count = len(state.missing_slots) if isinstance(state.missing_slots, list) else 0

        base_entities = self._build_search_entities(
            state.user_query,
            state.query_state,
            state.missing_slots,
        )
        routed_entities = self._normalize_entities(routed_args.get("entities", []))
        entities = list(base_entities)
        seen_entities = {str(item or "").strip().lower() for item in entities if str(item or "").strip()}
        for item in routed_entities:
            key = str(item or "").strip().lower()
            if not key or key in seen_entities:
                continue
            seen_entities.add(key)
            entities.append(str(item).strip())

        depth = routed_args.get("depth", 1)
        try:
            depth = max(1, min(int(depth), 3))
        except Exception:
            depth = 1
        if loop_state.consecutive_no_progress > 0 and state.missing_slots:
            depth = max(depth, 2)
            top_k = max(top_k, 8)
            focus_missing_slots = state.missing_slots[:2]
            if answer_type == "compute":
                focus_missing_slots = state.missing_slots[:8]
            entities = (
                self._build_search_entities(
                    "",
                    state.query_state,
                    focus_missing_slots,
                )
                + entities
            )

        required_count = len(self._required_slots(state.query_state))
        entity_cap = self._turn_entity_cap(answer_type, required_count, missing_count)
        entities = self._dedupe_preserve_order(entities, cap=entity_cap)
        search_entities = entities if routed_tool == "graph_search" else (entities or [state.user_query])
        return ToolSearchPlan(
            entities=entities,
            search_entities=search_entities,
            depth=depth,
            top_k=top_k,
        )

    @staticmethod
    def _update_consecutive_no_progress(
        loop_state: Any,
        new_entries_count: int,
        missing_slots: list[Any],
    ) -> None:
        if new_entries_count == 0 and missing_slots:
            loop_state.consecutive_no_progress += 1
        else:
            loop_state.consecutive_no_progress = 0

    async def _handle_retrieval_tool_call(
        self,
        *,
        state: AgentState,
        turn: int,
        tool_call_id: str,
        routed_tool: str,
        routed_args: dict[str, Any],
        loop_state: Any,
        top_k: int,
        started: float,
    ) -> None:
        plan = self._build_tool_search_plan(
            state=state,
            routed_tool=routed_tool,
            routed_args=routed_args,
            loop_state=loop_state,
            top_k=top_k,
        )

        txt, data, filter_diag = await self._entity_guarded_graph_search(
            entities=plan.search_entities,
            depth=plan.depth,
            top_k=plan.top_k,
            query_state=state.query_state,
            user_query=state.user_query,
        )
        data_raw_count = int(filter_diag.get("initial_raw", 0) or 0)
        data_raw_count += int(filter_diag.get("retry_raw", 0) or 0)
        loop_state.tool_calls_used += 1

        if txt:
            state.history.append({"role": "tool", "tool_call_id": tool_call_id, "name": routed_tool, "content": txt})
        state.all_context_data.extend(data)

        ledger_context = self._build_context_excerpt(
            data,
            query_state=state.query_state,
        )
        ledger_result = await self._extract_evidence_entries(
            state.query_state,
            ledger_context,
            filter_policy=state.filter_policy,
        )
        new_entries, model_missing_slots, ledger_debug = self._unpack_evidence_entries_result(
            ledger_result
        )
        prev_len = len(state.evidence_ledger)
        state.evidence_ledger = self._merge_ledger(state.evidence_ledger, new_entries)
        state.ledger_attempts.append({
            "step": f"ledger_update_{turn}",
            "retrieved": len(data),
            "diagnostics": ledger_debug,
        })
        state.missing_slots = self._resolve_missing_slots(
            state.query_state,
            state.evidence_ledger,
            model_missing_slots=model_missing_slots,
        )
        self._update_consecutive_no_progress(
            loop_state=loop_state,
            new_entries_count=len(new_entries),
            missing_slots=state.missing_slots,
        )

        await self._refresh_context_and_slots(state, trace_step=f"context_refresh_{turn}")
        append_trace(
            state.trace,
            step=f"ledger_update_{turn}",
            input={
                "entities": plan.entities,
                "depth": plan.depth,
                "top_k": plan.top_k,
            },
            output={
                "retrieved_raw": data_raw_count,
                "retrieved": len(data),
                "entity_filter": filter_diag,
                "new_entries": len(new_entries),
                "ledger_size_before": prev_len,
                "ledger_size_after": len(state.evidence_ledger),
                "model_missing_slots": model_missing_slots,
                "ledger_debug": ledger_debug,
                "missing_slots": state.missing_slots,
                "context_chars": len(state.context),
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _bootstrap_search_cap(answer_type: str, missing_count: int) -> int:
        if answer_type != "compute":
            return 8
        if missing_count >= 2:
            return 18
        return 12

    @staticmethod
    def _dedupe_preserve_order(items: list[str], cap: int) -> list[str]:
        deduped: list[str] = []
        seen = set()
        for item in items:
            key = str(item or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(item).strip())
            if len(deduped) >= cap:
                break
        return deduped

    def _build_bootstrap_entities(self, state: AgentState) -> list[str]:
        answer_type = str(state.query_state.get("answer_type", "") or "").strip().lower()
        query_entity = str(state.query_state.get("entity", "") or "").strip()
        bootstrap_entities: list[str] = []

        if query_entity and not self._is_generic_entity_label(query_entity):
            bootstrap_entities.extend(self._entity_search_aliases(query_entity)[:2])

        source_anchor = str(state.query_state.get("source_anchor") or "").strip()
        if source_anchor:
            bootstrap_entities.append(source_anchor)
            bootstrap_entities.extend(self._source_anchor_strict_markers(source_anchor)[:3])

        metric_text = self._normalize_metric_text(state.query_state.get("metric", ""))
        if metric_text:
            for alias in self._metric_alias_terms(metric_text)[:8]:
                alias_text = str(alias or "").strip().lower()
                if len(alias_text) < 4:
                    continue
                if alias_text in self._GENERIC_BOOTSTRAP_METRIC_TOKENS:
                    continue
                bootstrap_entities.append(alias_text)

        search_entities = self._build_search_entities(
            state.user_query,
            state.query_state,
            state.missing_slots,
        )
        missing_count = len(state.missing_slots) if isinstance(state.missing_slots, list) else 0
        search_cap = self._bootstrap_search_cap(answer_type, missing_count)
        bootstrap_entities.extend(search_entities[:search_cap])

        if state.user_query.strip() and answer_type != "compute":
            bootstrap_entities.append(state.user_query.strip())

        bootstrap_cap = 20 if answer_type == "compute" else 10
        return self._dedupe_preserve_order(bootstrap_entities, cap=bootstrap_cap)

    def _bootstrap_top_k(self, state: AgentState) -> int:
        top_k = min(max(RAGConfig.DEFAULT_TOP_K + 1, 4), 12)
        missing_count = len(state.missing_slots) if isinstance(state.missing_slots, list) else 0
        has_source_anchor = bool(str(state.query_state.get("source_anchor", "") or "").strip())
        if missing_count >= 2 or has_source_anchor:
            return max(top_k, 12)
        return top_k

    async def _bootstrap_hybrid_search(self, state: AgentState, loop_state: Any) -> None:
        """Initial hybrid retrieval pass to stabilize entry-node quality."""
        started = time.perf_counter()
        bootstrap_entities = self._build_bootstrap_entities(state)
        if not bootstrap_entities:
            return

        bootstrap_top_k = self._bootstrap_top_k(state)
        bootstrap_txt, bootstrap_data, bootstrap_filter_diag = await self._entity_guarded_graph_search(
            entities=bootstrap_entities,
            depth=1,
            top_k=bootstrap_top_k,
            query_state=state.query_state,
            user_query=state.user_query,
        )
        bootstrap_data_raw = int(bootstrap_filter_diag.get("initial_raw", 0) or 0)
        bootstrap_data_raw += int(bootstrap_filter_diag.get("retry_raw", 0) or 0)
        loop_state.tool_calls_used += 1
        if bootstrap_txt:
            state.history.append({
                "role": "tool",
                "tool_call_id": "bootstrap_0",
                "name": "graph_search_bootstrap",
                "content": bootstrap_txt,
            })
        state.all_context_data.extend(bootstrap_data)
        bootstrap_ledger_context = self._build_context_excerpt(
            bootstrap_data,
            query_state=state.query_state,
        )
        bootstrap_result = await self._extract_evidence_entries(
            state.query_state,
            bootstrap_ledger_context,
            filter_policy=state.filter_policy,
        )
        bootstrap_entries, bootstrap_missing_slots, bootstrap_ledger_debug = self._unpack_evidence_entries_result(
            bootstrap_result
        )
        state.evidence_ledger = self._merge_ledger(state.evidence_ledger, bootstrap_entries)
        state.ledger_attempts.append({
            "step": "bootstrap_retrieval",
            "retrieved": len(bootstrap_data),
            "diagnostics": bootstrap_ledger_debug,
        })
        state.missing_slots = self._resolve_missing_slots(
            state.query_state,
            state.evidence_ledger,
            model_missing_slots=bootstrap_missing_slots,
        )
        await self._refresh_context_and_slots(state, trace_step="context_refresh_bootstrap")
        append_trace(
            state.trace,
            step="bootstrap_retrieval",
            input={
                "entities": bootstrap_entities,
                "depth": 1,
                "top_k": bootstrap_top_k,
            },
            output={
                "retrieved_raw": bootstrap_data_raw,
                "retrieved": len(bootstrap_data),
                "entity_filter": bootstrap_filter_diag,
                "new_entries": len(bootstrap_entries),
                "model_missing_slots": bootstrap_missing_slots,
                "ledger_debug": bootstrap_ledger_debug,
                "missing_slots": state.missing_slots,
                "context_chars": len(state.context),
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
