from dataclasses import dataclass, field
from typing import Any

from models.hyporeflect.schemas import (
    ContextAtom,
    EvidenceEntry,
    FilterPolicy,
    QueryState,
    TraceEvent,
)


@dataclass(eq=False)
class AgentState:
    user_query: str
    history: list[dict[str, Any]]
    intent: str = "research"
    is_complex: bool = True
    plan: str = ""
    context: str = ""
    final_answer: str = ""
    all_context_data: list[dict[str, Any]] = field(default_factory=list)
    critique: str = ""
    query_state: QueryState = field(default_factory=dict)
    filter_policy: FilterPolicy = field(default_factory=dict)
    evidence_ledger: list[EvidenceEntry] = field(default_factory=list)
    evidence_atoms: list[ContextAtom] = field(default_factory=list)
    missing_slots: list[Any] = field(default_factory=list)
    ledger_attempts: list[dict[str, Any]] = field(default_factory=list)
    reflection_meta: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    # IDs of chunks already returned by any previous graph_search/retrieve
    # call within this query. Threaded through to subsequent calls so
    # bootstrap/turn-N retrievals do not re-surface the same chunk through
    # different seed paths (NEXT/HOP traversal can reach the same hub chunk
    # from many seeds, producing 30-50% duplication observed empirically).
    visited_chunk_ids: set[str] = field(default_factory=set)
