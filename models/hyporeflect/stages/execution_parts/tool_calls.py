import json
import logging
import re
import time
from typing import Any

from models.hyporeflect.state import AgentState
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


class ToolCallsSupport:
    def _route_tool_call(
        self,
        requested_tool: str,
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple:
        query = str(args.get("query", state.user_query) or state.user_query).strip()
        entities = self._normalize_entities(args.get("entities", []))
        expression = str(args.get("expression", "") or "").strip()
        precision = args.get("precision", None)
        depth_raw = args.get("depth", 2)
        try:
            depth = max(1, min(int(depth_raw), 4))
        except Exception:
            depth = 2

        if requested_tool == "calculator":
            return (
                "calculator",
                {"expression": expression, "precision": precision},
                "calculator_default",
            )

        fallback_entities = entities if entities else [query or state.user_query]
        if requested_tool == "graph_search":
            return "graph_search", {"entities": fallback_entities, "depth": depth}, "graph_search_default"
        if requested_tool == "retrieve":
            return "graph_search", {"entities": fallback_entities, "depth": depth}, "retrieve_redirected_to_graph_search"
        return "graph_search", {"entities": fallback_entities, "depth": depth}, "unknown_tool_redirected_to_graph_search"

    def _parse_tool_call_args(self, tool_call: Any) -> dict[str, Any]:
        try:
            return json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except Exception:
            logger.warning(
                "Failed to parse tool args for %s: %s",
                str(tool_call.function.name),
                tool_call.function.arguments,
            )
            return {}

    @staticmethod
    def _extract_textual_tool_calls(response_text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        if not response_text:
            return calls
        for raw_json in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response_text, flags=re.DOTALL):
            try:
                data = json.loads(raw_json)
            except Exception:
                continue
            name = str(data.get("name", "") or "").strip()
            arguments = data.get("arguments", {})
            if not name or not isinstance(arguments, dict):
                continue
            calls.append({"name": name, "arguments": arguments})
        return calls

    async def _handle_calculator_tool_call(
        self,
        *,
        state: AgentState,
        turn: int,
        tool_call_id: str,
        routed_args: dict[str, Any],
        loop_state: Any,
        started: float,
    ) -> None:
        calc = self._call_calculator(
            expression=routed_args.get("expression", ""),
            precision=routed_args.get("precision", None),
        )
        loop_state.tool_calls_used += 1

        tool_content = ""
        if calc.get("ok"):
            tool_content = (
                f"Calculator expression: {calc.get('expression')}\n"
                f"Calculator result: {calc.get('result')}"
            )
            calc_chunk = {
                "title": "CALCULATOR",
                "page": 0,
                "sent_id": -100000 - turn - loop_state.tool_calls_used,
                "text": tool_content,
            }
            state.all_context_data.append(calc_chunk)
        else:
            tool_content = (
                f"Calculator error for expression '{routed_args.get('expression', '')}': "
                f"{calc.get('error', 'unknown error')}"
            )

        state.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "calculator",
            "content": tool_content,
        })

        await self._refresh_context_and_slots(state, trace_step=f"context_refresh_{turn}")
        append_trace(
            state.trace,
            step=f"calculator_update_{turn}",
            input={
                "expression": routed_args.get("expression", ""),
                "precision": routed_args.get("precision", None),
            },
            output={
                "ok": bool(calc.get("ok", False)),
                "result": calc.get("result", ""),
                "error": calc.get("error", ""),
                "missing_slots": state.missing_slots,
                "context_chars": len(state.context),
            },
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _handle_tool_call_response(
        self,
        state: AgentState,
        turn: int,
        resp: Any,
        loop_state: Any,
    ) -> None:
        tool_calls = list(getattr(resp, "tool_calls", None) or [])
        if not tool_calls:
            return

        remaining_budget = max(loop_state.max_tool_calls - loop_state.tool_calls_used, 0)
        if remaining_budget <= 0:
            append_trace(
                state.trace,
                step=f"tool_calls_truncated_{turn}",
                input={
                    "received_tool_calls": len(tool_calls),
                    "processed_tool_calls": 0,
                    "max_tool_calls": loop_state.max_tool_calls,
                    "tool_calls_used": loop_state.tool_calls_used,
                },
                output={"reason": "tool_call_budget_exhausted"},
            )
            return

        tool_calls_to_process = tool_calls[:remaining_budget]

        for idx, tc in enumerate(tool_calls_to_process):
            started = time.perf_counter()
            function_obj = getattr(tc, "function", None)
            t_name = str(getattr(function_obj, "name", "") or "")
            args = self._parse_tool_call_args(tc)
            top_k = self._resolve_tool_top_k(state, args)
            tool_call_id = str(getattr(tc, "id", "") or f"auto_tool_{turn}_{idx}")

            routed_tool, routed_args, route_reason = self._route_tool_call(t_name, args, state)
            logger.info(
                "Tool routing: requested=%s -> routed=%s (%s)",
                t_name,
                routed_tool,
                route_reason,
            )

            if routed_tool == "calculator":
                await self._handle_calculator_tool_call(
                    state=state,
                    turn=turn,
                    tool_call_id=tool_call_id,
                    routed_args=routed_args,
                    loop_state=loop_state,
                    started=started,
                )
                continue

            await self._handle_retrieval_tool_call(
                state=state,
                turn=turn,
                tool_call_id=tool_call_id,
                routed_tool=routed_tool,
                routed_args=routed_args,
                loop_state=loop_state,
                top_k=top_k,
                started=started,
            )

        if len(tool_calls_to_process) < len(tool_calls):
            append_trace(
                state.trace,
                step=f"tool_calls_truncated_{turn}",
                input={
                    "received_tool_calls": len(tool_calls),
                    "processed_tool_calls": len(tool_calls_to_process),
                    "max_tool_calls": loop_state.max_tool_calls,
                    "tool_calls_used": loop_state.tool_calls_used,
                },
                output={"reason": "tool_call_budget_exhausted"},
            )

    async def _handle_textual_tool_call_response(
        self,
        state: AgentState,
        turn: int,
        response_text: str,
        loop_state: Any,
    ) -> bool:
        tool_calls = self._extract_textual_tool_calls(response_text)
        if not tool_calls:
            return False

        append_trace(
            state.trace,
            step=f"textual_tool_calls_parsed_{turn}",
            input=response_text,
            output={"count": len(tool_calls)},
        )
        remaining_budget = max(loop_state.max_tool_calls - loop_state.tool_calls_used, 0)
        if remaining_budget <= 0:
            append_trace(
                state.trace,
                step=f"tool_calls_truncated_{turn}",
                input={
                    "received_tool_calls": len(tool_calls),
                    "processed_tool_calls": 0,
                    "max_tool_calls": loop_state.max_tool_calls,
                    "tool_calls_used": loop_state.tool_calls_used,
                },
                output={"reason": "tool_call_budget_exhausted"},
            )
            return True

        for idx, tool_call in enumerate(tool_calls[:remaining_budget]):
            started = time.perf_counter()
            t_name = str(tool_call.get("name", "") or "")
            args = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments", {}), dict) else {}
            top_k = self._resolve_tool_top_k(state, args)
            tool_call_id = f"text_tool_{turn}_{idx}"

            routed_tool, routed_args, route_reason = self._route_tool_call(t_name, args, state)
            logger.info(
                "Textual tool routing: requested=%s -> routed=%s (%s)",
                t_name,
                routed_tool,
                route_reason,
            )

            if routed_tool == "calculator":
                await self._handle_calculator_tool_call(
                    state=state,
                    turn=turn,
                    tool_call_id=tool_call_id,
                    routed_args=routed_args,
                    loop_state=loop_state,
                    started=started,
                )
                continue

            await self._handle_retrieval_tool_call(
                state=state,
                turn=turn,
                tool_call_id=tool_call_id,
                routed_tool=routed_tool,
                routed_args=routed_args,
                loop_state=loop_state,
                top_k=top_k,
                started=started,
            )
        return True
