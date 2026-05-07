"""Execution stage core: ExecutionBase (constants + __init__), the
T_max=6 turn loop driver (RuntimeSupport.run), and ExpansionLoopState.

Paper §3.2.3 — Stage 3 Execution. The full :class:`ExecutionHandler` with
all support mixins is assembled in
:mod:`models.hyporeflect.stages.execution.__init__`. This module owns only
the loop-driving pieces; per-concern mixins live in sibling modules:

- planning_state.py — QueryState/Slot/Entity (paper §3.2.2 artifacts)
- search.py         — Search + tool call dispatch
- evidence.py       — Evidence ledger build / gate / rescue / audit
- context.py        — Context excerpt + atomization + packing + validation
- synthesis.py      — Final answer + forced retrace (lazy-answer guard)
- calculator.py     — Deterministic arithmetic
- overrides.py      — Domain post-synthesis overrides (FinanceBench-specific)
"""
import logging
import time
from dataclasses import dataclass
from typing import Any

from core.config import RAGConfig
from models.hyporeflect.state import AgentState
from models.hyporeflect.trace import append_trace
from utils.tool_definitions import get_all_tools


logger = logging.getLogger(__name__)


@dataclass
class ExpansionLoopState:
    tool_calls_used: int = 0
    max_tool_calls: int = 3
    consecutive_no_progress: int = 0


class ExecutionBase:
    """Class-level constants and ``__init__`` shared by ExecutionHandler.

    Kept here (rather than spread across the support mixins) so the static
    surface needed by other stages is co-located with the loop driver."""

    _VALID_ANSWER_TYPES = {"extract", "compute", "boolean", "list"}
    _VALID_MISSING_DATA_POLICIES = {
        "insufficient",
        "zero_if_not_explicit",
        "inapplicable_explain",
    }
    _VALID_SLOT_CONFLICT_STRATEGIES = {
        "best_supported",
        "keep_missing_on_tie",
    }
    _QUERY_STATEMENT_ANCHOR_TERMS = (
        "balance sheet",
        "statement of financial position",
        "income statement",
        "statement of income",
        "p&l",
        "cash flow statement",
        "statement of cash flows",
    )
    _OPEN_DOMAIN_ENTITY_STOPWORDS = {
        "what", "which", "who", "when", "where", "why", "how",
        "is", "are", "was", "were", "do", "does", "did",
        "among", "between", "compare", "comparison", "list", "name",
    }
    _INSUFFICIENT_ANSWER_MARKERS = (
        "insufficient evidence",
        "cannot be determined",
        "not possible to determine",
        "cannot answer with the available",
        "does not contain",
        "not available in the context",
    )
    _ZERO_POLICY_ANSWERS = {"@@answer: 0", "@@answer: 0.0", "@@answer: 0.00"}
    _CAPEX_RELAXED_GROUNDING_MARKERS = (
        "purchases of property",
        "capital expenditure",
        "capital expenditures",
        "pp&e",
    )
    _CAPEX_AMOUNT_MARKERS = (
        "capital expenditure",
        "capital expenditures",
        "purchases of property",
        "purchases of pp&e",
        "additions to property and equipment",
        "additions to pp&e",
        "property, plant and equipment",
        "property and equipment",
    )
    _CAPEX_RATIO_SPAN_MARKERS = (
        "as a percentage",
        "% of net revenues",
        "% of net revenue",
        "% of revenue",
        "percent of net revenues",
        "percent of revenue",
    )
    _GENERIC_BOOTSTRAP_METRIC_TOKENS = {
        "income", "capital", "assets", "liabilities", "shareholders",
        "shareowners", "total", "metric", "change", "business",
    }
    def __init__(self, llm, grag, stage_model: str = ""):
        self.llm = llm
        self.grag = grag
        self.stage_model = stage_model or RAGConfig.EXECUTION_MODEL
        self.context_node_budget = 36
        self.context_char_budget = 4200


class RuntimeSupport:
    """T_max=6 turn loop entry point (paper §3.2.3)."""

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
