from typing import Any, Literal, TypedDict


class SlotConstraint(TypedDict, total=False):
    entity: str
    period: str
    metric: str
    source_anchor: str
    unit: str
    rounding: str


class QueryState(TypedDict, total=False):
    entity: str
    period: str
    metric: str
    source_anchor: str | None
    answer_type: Literal["extract", "compute", "boolean", "list"]
    required_slots: list[SlotConstraint]
    unit: str | None
    rounding: str | None


class EvidenceEntry(TypedDict):
    slot: SlotConstraint | str
    value: str
    citation: str


class ContextAtom(TypedDict, total=False):
    atom_id: str
    citation: str
    span: str
    supports_slots: list[str]


class PackedContextResult(TypedDict, total=False):
    selected_atom_ids: list[str]
    slot_coverage: dict[str, Any]
    missing_slots: list[str]
    compressed_context: str


class ReflectionResult(TypedDict, total=False):
    decision: Literal["PASS", "FAIL"]
    issues: list[str]
    arithmetic_check: Literal["ok", "fail", "na"]


class FinalAnswerPayload(TypedDict):
    final_answer: str


class FilterPolicyMustMatch(TypedDict, total=False):
    entity: bool
    period: bool
    source_anchor: Literal["strict", "soft", "none"]


class FilterPolicy(TypedDict, total=False):
    must_match: FilterPolicyMustMatch
    preferred_markers: list[str]
    disallowed_patterns: list[str]
    slot_conflict_strategy: str


TraceEvent = dict[str, Any]
