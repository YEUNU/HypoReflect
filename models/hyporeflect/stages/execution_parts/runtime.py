import time
from dataclasses import dataclass
from typing import Any

from core.config import RAGConfig
from models.hyporeflect.state import AgentState
from models.hyporeflect.trace import append_trace
from utils.tool_definitions import get_all_tools


@dataclass
class ExpansionLoopState:
    tool_calls_used: int = 0
    max_tool_calls: int = 3
    consecutive_no_progress: int = 0


class RuntimeSupport:
    async def run(self, state: AgentState):
        await self._initialize_query_state_phase(state)
        loop_state = ExpansionLoopState()
        await self._bootstrap_hybrid_search(state, loop_state)

        for turn in range(RAGConfig.MAX_AGENT_TURNS):
            if self._should_terminate_expansion(state, loop_state):
                break

            messages = self._build_expansion_messages(state)
            turn_started = time.perf_counter()
            resp_kwargs: dict[str, Any] = dict(tools=get_all_tools(), apply_default_sampling=False)
            if self.stage_model:
                resp_kwargs["model"] = self.stage_model
            resp = await self.llm.generate_response(messages, **resp_kwargs)
            append_trace(
                state.trace,
                step=f"execution_turn_{turn}",
                input=messages,
                output=str(resp),
                duration_ms=(time.perf_counter() - turn_started) * 1000.0,
            )

            if hasattr(resp, "tool_calls") and resp.tool_calls:
                await self._handle_tool_call_response(state, turn, resp, loop_state)
                continue
            if await self._handle_textual_tool_call_response(state, turn, str(resp), loop_state):
                continue
            if self._handle_direct_response(state, turn, resp):
                break

        await self._run_forced_synthesis_if_needed(state)
