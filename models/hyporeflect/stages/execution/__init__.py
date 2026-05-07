"""Stage 3: Execution (paper §3.2.3).

ExecutionHandler is composed from per-concern mixins; a single line per
mixin makes the paper mapping obvious. The MRO order matches the previous
``execution_parts`` chain so behaviour is unchanged.
"""
from .calculator import CalculatorSupport
from .context import ContextSupport, ResidualSupport
from .evidence import EvidenceSupport
from .handler import ExecutionBase, ExpansionLoopState, RuntimeSupport
from .planning_state import EntitySupport, QueryStateSupport, SlotSupport
from .search import SearchSupport, ToolCallsSupport, ToolSearchPlan
from .synthesis import ForcedSynthesisSupport, SynthesisSupport


class ExecutionHandler(
    ResidualSupport,
    QueryStateSupport,
    SlotSupport,
    EntitySupport,
    CalculatorSupport,
    ContextSupport,
    EvidenceSupport,
    ToolCallsSupport,
    SearchSupport,
    SynthesisSupport,
    ForcedSynthesisSupport,
    RuntimeSupport,
    ExecutionBase,
):
    pass


__all__ = [
    "ExecutionHandler",
    "ExpansionLoopState",
    "ToolSearchPlan",
]
