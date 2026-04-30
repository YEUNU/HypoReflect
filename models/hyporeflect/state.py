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
