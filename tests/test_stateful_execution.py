from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import RAGConfig
from models.hyporeflect.service import AgentService
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.execution import ExecutionHandler, ExpansionLoopState
from models.hyporeflect.stages.reflection import ReflectionHandler


def test_compute_missing_slots_from_ledger():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "required_slots": ["revenue", "capex"],
    }
    ledger = [
        {"slot": "revenue", "value": "100", "citation": "[[A, Page 1, Chunk 1]]"},
    ]
    missing = handler._compute_missing_slots(query_state, ledger)  # noqa: SLF001
    assert missing == ["capex"]


def test_compute_missing_slots_best_supported_treats_conflict_as_covered():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "required_slots": ["revenue"],
    }
    ledger = [
        {"slot": "revenue", "value": "2.5 billion", "citation": "[[A, Page 1, Chunk 1]]"},
        {"slot": "revenue", "value": "2,500,000,000", "citation": "[[A, Page 1, Chunk 2]]"},
    ]

    missing = handler._compute_missing_slots(  # noqa: SLF001
        query_state,
        ledger,
        slot_conflict_strategy="best_supported",
    )
    assert missing == []


def test_compute_missing_slots_keep_missing_on_tie_marks_missing():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "required_slots": ["revenue"],
    }
    ledger = [
        {"slot": "revenue", "value": "100", "citation": "[[A, Page 1, Chunk 1]]"},
        {"slot": "revenue", "value": "130", "citation": "[[A, Page 1, Chunk 2]]"},
    ]

    missing = handler._compute_missing_slots(  # noqa: SLF001
        query_state,
        ledger,
        slot_conflict_strategy="keep_missing_on_tie",
    )
    assert missing == ["revenue"]


def test_value_grounded_in_span_accepts_unit_scaled_equivalence():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    assert handler._value_grounded_in_span("2.5 billion", "$2,500,000,000") is True  # noqa: SLF001
    assert handler._value_grounded_in_span("(123)", "-123") is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_execution_blocks_insufficient_when_slots_are_grounded():
    llm = AsyncMock()
    grag = MagicMock()
    handler = ExecutionHandler(llm=llm, grag=grag)
    state = AgentState("What is A FY2022 net income?", [])

    handler._infer_query_state = AsyncMock(return_value={  # noqa: SLF001
        "entity": "A",
        "period": "FY2022",
        "metric": "net income",
        "answer_type": "extract",
        "required_slots": ["net income"],
        "unit": None,
        "rounding": None,
    })
    handler._extract_evidence_entries = AsyncMock(return_value=(  # noqa: SLF001
        [{"slot": "net income", "value": "100", "citation": "[[A_2022_10K, Page 1, Chunk 1]]"}],
        [],
    ))

    tc = MagicMock(id="tc1")
    tc.function.name = "graph_search"
    tc.function.arguments = '{"entities": ["A FY2022 net income"], "depth": 1, "top_k": 3}'

    llm.generate_response = AsyncMock(side_effect=[
        "@@ANSWER: 100 [[A_2022_10K, Page 1, Chunk 1]]",
    ])
    grag.graph_search = AsyncMock(return_value=(
        "[[A_2022_10K, Page 1, Chunk 1]]\nNet income was 100.",
        [{"title": "A_2022_10K", "page": 1, "sent_id": 1, "text": "Net income was 100."}],
    ))

    await handler.run(state)

    assert "insufficient evidence" not in state.final_answer.lower()
    assert "[[A_2022_10K, Page 1, Chunk 1]]" in state.final_answer


@pytest.mark.asyncio
async def test_llm_gate_entries_soft_fallback_recovers_coverage():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={
        "decisions": [
            {"index": 0, "keep": False, "reason": "reject all"},
            {"index": 1, "keep": False, "reason": "reject all"},
        ]
    })
    handler = ExecutionHandler(llm=llm, grag=MagicMock())

    query_state = {
        "answer_type": "extract",
        "required_slots": ["slot_a", "slot_b"],
    }
    entries = [
        {"slot": "slot_a", "value": "10", "citation": "[[DOC_A, Page 1, Chunk 1]]"},
        {"slot": "slot_b", "value": "20", "citation": "[[DOC_A, Page 2, Chunk 2]]"},
    ]
    context = (
        "[[DOC_A, Page 1, Chunk 1]]\nA=10\n\n"
        "[[DOC_A, Page 2, Chunk 2]]\nB=20"
    )

    kept, diag = await handler._llm_gate_entries(query_state, context, entries)  # noqa: SLF001

    assert len(kept) >= 1
    assert diag.get("fallback_applied") is True
    assert diag.get("coverage_regressed_slots", 0) >= 1


@pytest.mark.asyncio
async def test_forced_synthesis_prefers_calculator_direct_answer_for_numeric_compute():
    llm = AsyncMock()
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    handler._compute_with_calculator_from_ledger = AsyncMock(return_value={  # noqa: SLF001
        "ok": True,
        "expression": "200/100",
        "result": "2",
    })
    handler._generate_single_final_answer = AsyncMock(return_value=(  # noqa: SLF001
        "@@ANSWER: should_not_be_used",
        [{"attempt": 1, "accepted": True, "reason": "mock"}],
    ))

    state = AgentState("What is the FY2022 ratio?", [])
    state.query_state = {
        "answer_type": "compute",
        "metric": "ratio",
        "required_slots": ["slot_a", "slot_b"],
    }
    state.missing_slots = []
    state.evidence_ledger = [
        {"slot": "slot_a", "value": "200", "citation": "[[DOC_A, Page 1, Chunk 1]]"},
        {"slot": "slot_b", "value": "100", "citation": "[[DOC_A, Page 2, Chunk 2]]"},
    ]
    state.context = (
        "[[DOC_A, Page 1, Chunk 1]]\nValue A = 200\n\n"
        "[[DOC_A, Page 2, Chunk 2]]\nValue B = 100"
    )

    await handler._run_forced_synthesis_if_needed(state)  # noqa: SLF001

    assert "@@ANSWER: 2" in state.final_answer
    assert handler._generate_single_final_answer.await_count == 0  # noqa: SLF001
    steps = [item.get("step") for item in state.trace if isinstance(item, dict)]
    assert "execution_compute_direct_answer" in steps
    assert "execution_forced_synthesis" in steps


@pytest.mark.asyncio
async def test_forced_synthesis_realigns_collapsed_multi_period_slots_before_calculator():
    llm = AsyncMock()
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    handler._compute_with_calculator_from_ledger = AsyncMock(return_value={  # noqa: SLF001
        "ok": True,
        "expression": "24.6 - 21.6",
        "result": "3.0",
    })
    handler._generate_single_final_answer = AsyncMock(return_value=(  # noqa: SLF001
        "@@ANSWER: should_not_be_used",
        [{"attempt": 1, "accepted": True, "reason": "mock"}],
    ))

    state = AgentState("How much has the effective tax rate changed between FY2021 and FY2022?", [])
    slot_2021 = {
        "entity": "american express",
        "period": "fy2021",
        "metric": "effective tax rate",
    }
    slot_2022 = {
        "entity": "american express",
        "period": "fy2022",
        "metric": "effective tax rate",
    }
    state.query_state = {
        "answer_type": "compute",
        "metric": "effective tax rate change",
        "required_slots": [slot_2021, slot_2022],
    }
    state.missing_slots = []
    state.evidence_ledger = [
        {
            "slot": slot_2021,
            "value": "21.6",
            "citation": "[[AMERICANEXPRESS_2022_10K, Page 47, Chunk 224]]",
        },
        {
            "slot": slot_2022,
            "value": "21.6",
            "citation": "[[AMERICANEXPRESS_2022_10K, Page 47, Chunk 224]]",
        },
    ]
    state.context = (
        "[[AMERICANEXPRESS_2022_10K, Page 47, Chunk 224]]\n"
        "In 2022, the effective tax rate was 21.6%.\n\n"
        "[[AMERICANEXPRESS_2022_10K, Page 47, Chunk 225]]\n"
        "In 2021, the effective tax rate was 24.6%."
    )
    state.all_context_data = [
        {
            "title": "AMERICANEXPRESS_2022_10K",
            "page": 47,
            "sent_id": 224,
            "text": "In 2022, the effective tax rate was 21.6%.",
        },
        {
            "title": "AMERICANEXPRESS_2022_10K",
            "page": 47,
            "sent_id": 225,
            "text": "In 2021, the effective tax rate was 24.6%.",
        },
    ]

    await handler._run_forced_synthesis_if_needed(state)  # noqa: SLF001

    assert "@@ANSWER: 3" in state.final_answer
    steps = [item for item in state.trace if isinstance(item, dict)]
    realign = [item for item in steps if item.get("step") == "execution_compute_slot_realign"]
    assert realign
    assert bool(realign[-1].get("output", {}).get("applied")) is True
    values = {str(entry.get("value", "")) for entry in state.evidence_ledger}
    assert "24.6" in values
    assert "21.6" in values


@pytest.mark.asyncio
async def test_reflection_fails_when_numeric_compute_answer_mismatches_calculator_result():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={
        "decision": "PASS",
        "arithmetic_check": "na",
        "issues": [],
    })
    reflection = ReflectionHandler(llm)
    state = AgentState("What is the FY2022 ratio?", [])
    state.query_state = {
        "answer_type": "compute",
        "metric": "ratio",
        "required_slots": ["slot_a", "slot_b"],
    }
    state.missing_slots = []
    state.final_answer = "@@ANSWER: 3 [[DOC_A, Page 1, Chunk 1]]"
    state.context = "[[DOC_A, Page 1, Chunk 1]]\nRatio section"
    state.trace = [
        {
            "step": "execution_compute_tool",
            "output": {"ok": True, "result": "2"},
        }
    ]

    passed = await reflection.run(state)

    assert passed is False
    assert "deterministic arithmetic check failed" in state.critique.lower()
    assert state.reflection_meta.get("arithmetic_check") == "fail"


@pytest.mark.asyncio
async def test_entity_guarded_graph_search_compute_relaxed_fallback_uses_alias_and_keeps_nodes():
    handler = ExecutionHandler(llm=AsyncMock(), grag=MagicMock())
    nodes = [
        {
            "title": "NOTE_SUMMARY",
            "page": 1,
            "sent_id": 1,
            "text": "ACTIVISIONBLIZZARD property and equipment - net for FY2019.",
        }
    ]
    handler._call_graph_search = AsyncMock(side_effect=[("initial", nodes), ("retry", nodes)])  # noqa: SLF001

    txt, data, diagnostics = await handler._entity_guarded_graph_search(  # noqa: SLF001
        entities=["Activision Blizzard FY2019 fixed asset turnover"],
        depth=2,
        top_k=8,
        query_state={
            "entity": "Activision Blizzard",
            "period": "FY2019",
            "metric": "fixed asset turnover ratio",
            "answer_type": "compute",
        },
        user_query="What is Activision Blizzard FY2019 fixed asset turnover ratio?",
    )

    assert txt
    assert len(data) == 1
    assert diagnostics.get("initial_kept") == 0
    assert diagnostics.get("retry_used") is True
    assert diagnostics.get("relaxed_used") is True
    retry_call = handler._call_graph_search.await_args_list[1]  # noqa: SLF001
    retry_entities = [str(x).lower() for x in retry_call.args[0]]
    assert "activisionblizzard" in retry_entities


@pytest.mark.asyncio
async def test_entity_guarded_graph_search_non_compute_relaxed_fallback_when_empty():
    handler = ExecutionHandler(llm=AsyncMock(), grag=MagicMock())
    nodes = [
        {
            "title": "NOTE_SUMMARY",
            "page": 1,
            "sent_id": 1,
            "text": "ACTIVISIONBLIZZARD property and equipment - net for FY2019.",
        }
    ]
    handler._call_graph_search = AsyncMock(side_effect=[("initial", nodes), ("retry", nodes)])  # noqa: SLF001

    _, data, diagnostics = await handler._entity_guarded_graph_search(  # noqa: SLF001
        entities=["Activision Blizzard FY2019 pp&e"],
        depth=1,
        top_k=6,
        query_state={
            "entity": "Activision Blizzard",
            "period": "FY2019",
            "metric": "net pp&e",
            "answer_type": "extract",
        },
        user_query="What is Activision Blizzard FY2019 net pp&e?",
    )

    assert len(data) == 1
    assert diagnostics.get("retry_used") is True
    assert diagnostics.get("relaxed_used") is True


@pytest.mark.asyncio
async def test_entity_guarded_graph_search_compute_retries_on_sparse_year_coverage():
    handler = ExecutionHandler(llm=AsyncMock(), grag=MagicMock())
    initial_nodes = [
        {
            "title": "ACTIVISIONBLIZZARD_2017_10K",
            "page": 94,
            "sent_id": 1,
            "text": "Capital expenditures was (155) for FY2017.",
        }
    ]
    retry_nodes = [
        {
            "title": "ACTIVISIONBLIZZARD_2019_10K",
            "page": 44,
            "sent_id": 2,
            "text": "Total net revenues was 6489 for FY2019.",
        }
    ]
    handler._call_graph_search = AsyncMock(  # noqa: SLF001
        side_effect=[("initial", initial_nodes), ("retry", retry_nodes)]
    )

    _, data, diagnostics = await handler._entity_guarded_graph_search(  # noqa: SLF001
        entities=["Activision Blizzard FY2017-FY2019 capex revenue"],
        depth=2,
        top_k=8,
        query_state={
            "entity": "Activision Blizzard",
            "period": "FY2017-FY2019",
            "metric": "capex as a % of revenue",
            "answer_type": "compute",
        },
        user_query="What is Activision Blizzard FY2017-FY2019 capex as a % of revenue?",
    )

    assert diagnostics.get("initial_kept") == 1
    assert diagnostics.get("retry_used") is True
    assert diagnostics.get("retry_reason") == "compute_sparse_year_coverage"
    assert diagnostics.get("retry_kept") == 1
    assert len(data) == 1
    assert data[0].get("title") == "ACTIVISIONBLIZZARD_2019_10K"
    assert handler._call_graph_search.await_count == 2  # noqa: SLF001


@pytest.mark.asyncio
async def test_extract_evidence_entries_rejects_cashflow_anchor_mismatch():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={
        "entries": [
            {
                "slot": {
                    "entity": "3m",
                    "period": "fy2018",
                    "metric": "capital expenditure",
                    "source_anchor": "cash flow statement",
                },
                "value": "75",
                "citation": "[[3M_2018_10K, Page 7, Chunk 35]]",
            }
        ],
        "missing_slots": [],
    })
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "FY2018",
        "metric": "capital expenditure",
        "source_anchor": "cash flow statement",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2018",
                "metric": "capital expenditure",
                "source_anchor": "cash flow statement",
            }
        ],
    }
    context = (
        "[[3M_2018_10K, Page 7, Chunk 35]]\n"
        "Environmental expenditures for capital projects related to protecting the environment."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={},
    )

    assert entries == []
    assert missing_slots
    reject_reasons = diagnostics.get("reject_reasons", {})
    assert (
        reject_reasons.get("source_anchor_mismatch", 0) >= 1
        or reject_reasons.get("value_not_in_cited_span", 0) >= 1
    )


@pytest.mark.asyncio
async def test_extract_evidence_entries_invokes_slot_rescue_on_zero_accepted_with_rich_context():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={"entries": [], "missing_slots": []})
    handler = ExecutionHandler(llm=llm, grag=MagicMock())

    async def _passthrough_gate(query_state, context_excerpt, entries):
        return entries, {
            "checked": len(entries),
            "kept": len(entries),
            "rejected": 0,
            "reject_reasons": {},
            "fallback_applied": False,
            "fallback_kept": 0,
            "coverage_regressed_slots": 0,
        }

    handler._llm_gate_entries = AsyncMock(side_effect=_passthrough_gate)  # noqa: SLF001
    handler._rescue_missing_slot_entries = AsyncMock(  # noqa: SLF001
        return_value=[
            {
                "slot": {
                    "entity": "3m",
                    "period": "fy2022",
                    "metric": "revenue",
                    "source_anchor": "income statement",
                },
                "value": "100",
                "citation": "[[3M_2022_10K, Page 10, Chunk 1]]",
            }
        ]
    )
    query_state = {
        "entity": "3M",
        "period": "FY2022",
        "metric": "revenue",
        "source_anchor": "income statement",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2022",
                "metric": "revenue",
                "source_anchor": "income statement",
            }
        ],
    }
    context = (
        "[[3M_2022_10K, Page 10, Chunk 1]]\nConsolidated statement of income. Revenue was 100.\n\n"
        "[[3M_2022_10K, Page 11, Chunk 2]]\nConsolidated statement of income. Revenue was 98.\n\n"
        "[[3M_2022_10K, Page 12, Chunk 3]]\nConsolidated statement of income. Operating income was 30.\n\n"
        "[[3M_2022_10K, Page 13, Chunk 4]]\nConsolidated statement of income. Net income was 22."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={},
    )

    assert entries
    assert missing_slots == []
    assert diagnostics.get("slot_rescue_invoked") is True
    assert handler._rescue_missing_slot_entries.await_count >= 1  # noqa: SLF001


def test_should_force_missing_slot_rescue_compute_with_partial_entries():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    should_rescue = handler._should_force_missing_slot_rescue(  # noqa: SLF001
        query_state={"answer_type": "compute", "required_slots": ["a", "b"]},
        model_missing_slots=["b"],
        diagnostics={"accepted_entries": 1, "reject_reasons": {}},
        context_citation_count=3,
    )
    assert should_rescue is True


@pytest.mark.asyncio
async def test_extract_evidence_entries_keep_missing_on_tie_conflict_strategy():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(
        return_value={
            "entries": [
                {
                    "slot": {
                        "entity": "3m",
                        "period": "fy2022",
                        "metric": "revenue",
                        "source_anchor": "income statement",
                    },
                    "value": "100",
                    "citation": "[[3M_2022_10K, Page 10, Chunk 1]]",
                },
                {
                    "slot": {
                        "entity": "3m",
                        "period": "fy2022",
                        "metric": "revenue",
                        "source_anchor": "income statement",
                    },
                    "value": "130",
                    "citation": "[[3M_2022_10K, Page 11, Chunk 2]]",
                },
            ],
            "missing_slots": [],
        }
    )
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "FY2022",
        "metric": "revenue",
        "source_anchor": "income statement",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2022",
                "metric": "revenue",
                "source_anchor": "income statement",
            }
        ],
    }
    context = (
        "[[3M_2022_10K, Page 10, Chunk 1]]\nConsolidated statement of income. Revenue 2022 was 100.\n\n"
        "[[3M_2022_10K, Page 11, Chunk 2]]\nConsolidated statement of income. Revenue 2022 was 130."
    )
    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={"slot_conflict_strategy": "keep_missing_on_tie"},
    )

    assert entries == []
    assert missing_slots
    assert diagnostics.get("slot_conflict_strategy") == "keep_missing_on_tie"
    assert diagnostics.get("reject_reasons", {}).get("slot_conflict_keep_missing", 0) >= 1


@pytest.mark.asyncio
async def test_extract_evidence_entries_compute_conflict_keeps_local_best_supported():
    llm = AsyncMock()
    slot = {
        "entity": "amcor",
        "period": "fy2023",
        "metric": "sales",
        "source_anchor": "income statement",
    }
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    llm.generate_json = AsyncMock(
        return_value={
            "entries": [
                {
                    "slot": slot,
                    "value": "30",
                    "citation": "[[AMCOR_2023_10K, Page 29, Chunk 120]]",
                },
                {
                    "slot": slot,
                    "value": "14,694",
                    "citation": "[[AMCOR_2023_10K, Page 50, Chunk 219]]",
                },
            ],
            "missing_slots": [],
        }
    )
    query_state = {
        "entity": "AMCOR",
        "period": "FY2023",
        "metric": "sales",
        "source_anchor": "income statement",
        "answer_type": "compute",
        "required_slots": [slot],
    }
    context = (
        "[[AMCOR_2023_10K, Page 29, Chunk 120]]\n"
        "In 2023, net sales increased by $30 million.\n\n"
        "[[AMCOR_2023_10K, Page 50, Chunk 219]]\n"
        "In 2023, net sales were $14,694 million."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={"slot_conflict_strategy": "best_supported"},
    )

    assert missing_slots == []
    assert len(entries) == 1
    assert str(entries[0].get("value", "")) == "30"
    assert diagnostics.get("reject_reasons", {}).get("slot_conflict_best_supported", 0) >= 1
    assert diagnostics.get("slot_conflict_verifier_checked", 0) == 0


@pytest.mark.asyncio
async def test_extract_evidence_entries_compute_conflict_no_unresolved_without_candidate_verifier():
    llm = AsyncMock()
    slot = {
        "entity": "amcor",
        "period": "fy2023",
        "metric": "sales",
        "source_anchor": "income statement",
    }
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    llm.generate_json = AsyncMock(
        return_value={
            "entries": [
                {
                    "slot": slot,
                    "value": "30",
                    "citation": "[[AMCOR_2023_10K, Page 29, Chunk 120]]",
                },
                {
                    "slot": slot,
                    "value": "14,694",
                    "citation": "[[AMCOR_2023_10K, Page 50, Chunk 219]]",
                },
            ],
            "missing_slots": [],
        }
    )
    query_state = {
        "entity": "AMCOR",
        "period": "FY2023",
        "metric": "sales",
        "source_anchor": "income statement",
        "answer_type": "compute",
        "required_slots": [slot],
    }
    context = (
        "[[AMCOR_2023_10K, Page 29, Chunk 120]]\n"
        "In 2023, net sales increased by $30 million.\n\n"
        "[[AMCOR_2023_10K, Page 50, Chunk 219]]\n"
        "In 2023, net sales were $14,694 million."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={"slot_conflict_strategy": "best_supported"},
    )

    assert len(entries) == 1
    assert missing_slots == []
    assert diagnostics.get("reject_reasons", {}).get("slot_conflict_unresolved", 0) == 0
    assert diagnostics.get("slot_conflict_verifier_unresolved", 0) == 0


@pytest.mark.asyncio
async def test_extract_evidence_entries_rejects_capex_without_capex_marker():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(
        return_value={
            "entries": [
                {
                    "slot": {
                        "entity": "3m",
                        "period": "fy2018",
                        "metric": "capital expenditure",
                        "source_anchor": "cash flow statement",
                    },
                    "value": "311",
                    "citation": "[[3M_2018_10K, Page 7, Chunk 35]]",
                }
            ],
            "missing_slots": [],
        }
    )
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "FY2018",
        "metric": "capital expenditure",
        "source_anchor": "cash flow statement",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2018",
                "metric": "capital expenditure",
                "source_anchor": "cash flow statement",
            }
        ],
    }
    context = (
        "[[3M_2018_10K, Page 7, Chunk 35]]\n"
        "Amortization of capitalized software development costs was 311."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={},
    )

    assert entries == []
    assert missing_slots
    reject_reasons = diagnostics.get("reject_reasons", {})
    assert (
        reject_reasons.get("capex_marker_missing", 0) >= 1
        or reject_reasons.get("source_anchor_mismatch", 0) >= 1
    )


@pytest.mark.asyncio
async def test_extract_evidence_entries_rejects_compute_value_not_near_metric_term():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(
        return_value={
            "entries": [
                {
                    "slot": {
                        "entity": "coca cola",
                        "period": "fy2022",
                        "metric": "net income attributable to shareholders",
                        "source_anchor": "income statement",
                    },
                    "value": "31",
                    "citation": "[[COCACOLA_2022_10K, Page 141, Chunk 1015]]",
                }
            ],
            "missing_slots": [],
        }
    )
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "Coca Cola",
        "period": "FY2022",
        "metric": "dividend payout ratio",
        "source_anchor": "income statement",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "coca cola",
                "period": "fy2022",
                "metric": "net income attributable to shareholders",
                "source_anchor": "income statement",
            }
        ],
    }
    context = (
        "[[COCACOLA_2022_10K, Page 141, Chunk 1015]]\n"
        "The following financial information includes consolidated statements of income "
        "for the years ended December 31, 2022, 2021 and 2020."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={},
    )

    assert entries == []
    assert missing_slots
    assert diagnostics.get("reject_reasons", {}).get("value_not_near_metric_term", 0) >= 1


@pytest.mark.asyncio
async def test_rescue_missing_slot_entries_compute_uses_deterministic_fallback_when_empty():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={"entries": [], "missing_slots": []})
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "FY2022",
        "metric": "revenue",
        "source_anchor": "income statement",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2022",
                "metric": "revenue",
                "source_anchor": "income statement",
            }
        ],
    }
    missing_slots = [
        {
            "entity": "3m",
            "period": "fy2022",
            "metric": "revenue",
            "source_anchor": "income statement",
        }
    ]
    context = (
        "[[3M_2022_10K, Page 10, Chunk 1]]\n"
        "Consolidated statement of income. Total net revenues for 2022 were 34,229."
    )

    entries = await handler._rescue_missing_slot_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        missing_slots=missing_slots,
    )

    assert len(entries) == 1
    assert entries[0].get("citation") == "[[3M_2022_10K, Page 10, Chunk 1]]"
    assert str(entries[0].get("value", "")).strip() != ""


def test_verify_answer_grounding_rejects_boolean_without_yes_or_no():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "answer_type": "boolean",
        "missing_data_policy": "insufficient",
    }
    answer = "@@ANSWER: 6.00 [[DOC_A, Page 1, Chunk 1]]"
    context = "[[DOC_A, Page 1, Chunk 1]]\nDividend per share was 6.00."

    ok, reason = handler._verify_answer_grounding(  # noqa: SLF001
        answer=answer,
        query_state=query_state,
        evidence_ledger=[],
        context=context,
        missing_slots=[],
    )

    assert ok is False
    assert "boolean answer must start with yes/no" in reason


def test_build_search_entities_keeps_multi_slot_terms_for_non_compute():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "FY2022",
        "metric": "capital intensity",
        "answer_type": "boolean",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2022",
                "metric": "capital expenditures",
                "source_anchor": "cash flow statement",
            },
            {
                "entity": "3m",
                "period": "fy2022",
                "metric": "revenue",
                "source_anchor": "income statement",
            },
        ],
    }

    entities = handler._build_search_entities(  # noqa: SLF001
        query="Is 3M a capital-intensive business based on FY2022 data?",
        query_state=query_state,
        missing_slots=query_state["required_slots"],
    )
    lowered = [str(item).lower() for item in entities]
    assert any("capital expenditures" in item for item in lowered)
    assert any("revenue" in item for item in lowered)


def test_build_search_entities_compute_keeps_multi_year_slot_periods():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "entity": "Activision Blizzard",
        "period": "FY2017-FY2019",
        "metric": "capex as a % of revenue",
        "answer_type": "compute",
        "source_anchor": "cash flow statement",
        "required_slots": [
            {
                "entity": "activision blizzard",
                "period": "fy2017",
                "metric": "capex",
                "source_anchor": "cash flow statement",
            },
            {
                "entity": "activision blizzard",
                "period": "fy2018",
                "metric": "capex",
                "source_anchor": "cash flow statement",
            },
            {
                "entity": "activision blizzard",
                "period": "fy2019",
                "metric": "capex",
                "source_anchor": "cash flow statement",
            },
            {
                "entity": "activision blizzard",
                "period": "fy2017",
                "metric": "revenue",
                "source_anchor": "income statement",
            },
            {
                "entity": "activision blizzard",
                "period": "fy2018",
                "metric": "revenue",
                "source_anchor": "income statement",
            },
            {
                "entity": "activision blizzard",
                "period": "fy2019",
                "metric": "revenue",
                "source_anchor": "income statement",
            },
        ],
    }

    entities = handler._build_search_entities(  # noqa: SLF001
        query=(
            "What is the FY2017 - FY2019 3 year average of capex as a % of revenue "
            "for Activision Blizzard?"
        ),
        query_state=query_state,
        missing_slots=query_state["required_slots"],
    )
    lowered = [str(item).lower() for item in entities]
    assert "fy2017" in lowered
    assert "fy2018" in lowered
    assert "fy2019" in lowered
    assert any("fy2019 capex" in item or "fy2019 revenue" in item for item in lowered)


def test_normalize_final_answer_for_query_capex_flips_negative_outflow():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "answer_type": "extract",
        "metric": "capital expenditure",
        "source_anchor": "cash flow statement",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2018",
                "metric": "capital expenditure",
                "source_anchor": "cash flow statement",
            }
        ],
    }
    answer = "@@ANSWER: -1577 [[3M_2018_10K, Page 59, Chunk 365]]"

    normalized = handler._normalize_final_answer_for_query(answer, query_state)  # noqa: SLF001

    assert normalized.startswith("@@ANSWER: 1577 ")
    assert "[[3M_2018_10K, Page 59, Chunk 365]]" in normalized


def test_apply_query_state_heuristics_promotes_boolean_and_list():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())

    boolean_qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query="Does 3M maintain a stable trend of dividend distribution?",
        query_state={"answer_type": "extract"},
    )
    list_qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query="Which debt securities are registered to trade on a national securities exchange?",
        query_state={"answer_type": "extract"},
    )

    assert boolean_qs.get("answer_type") == "boolean"
    assert list_qs.get("answer_type") == "list"


def test_apply_query_state_heuristics_adds_driver_slot():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query="What drove operating margin change as of FY2022 for 3M?",
        query_state={
            "entity": "3M",
            "period": "FY2022",
            "metric": "operating margin",
            "answer_type": "extract",
            "required_slots": [
                {"entity": "3m", "period": "fy2022", "metric": "operating margin"},
            ],
        },
    )
    slots = qs.get("required_slots", [])
    assert isinstance(slots, list)
    assert any("change drivers" in str((slot or {}).get("metric", "")) for slot in slots if isinstance(slot, dict))


def test_apply_query_state_heuristics_relaxes_dividend_anchor():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query="Does 3M maintain a stable trend of dividend distribution?",
        query_state={
            "entity": "3M",
            "metric": "dividend distribution",
            "answer_type": "boolean",
            "source_anchor": "cash flow statement",
            "required_slots": [
                {
                    "entity": "3m",
                    "metric": "dividend distribution",
                    "source_anchor": "cash flow statement",
                }
            ],
        },
    )
    assert qs.get("source_anchor") is None
    slot = qs.get("required_slots", [])[0]
    assert "source_anchor" not in slot


def test_apply_query_state_heuristics_backfills_explicit_anchor_to_slots():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query=(
            "What is 3M's FY2022 net AR? "
            "Use only values shown in the balance sheet."
        ),
        query_state={
            "entity": "3M",
            "period": "FY2022",
            "metric": "net AR",
            "answer_type": "extract",
            "required_slots": [
                {"entity": "3m", "period": "fy2022", "metric": "net ar"},
            ],
        },
    )
    assert qs.get("source_anchor") == "balance sheet"
    slot = qs.get("required_slots", [])[0]
    assert slot.get("source_anchor") == "balance sheet"


def test_apply_query_state_heuristics_maps_multi_statement_anchors_by_slot_metric():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query=(
            "Calculate capex as a % of revenue for FY2022 using the P&L statement "
            "and the statement of cash flows."
        ),
        query_state={
            "entity": "3M",
            "period": "FY2022",
            "metric": "capex as a % of revenue",
            "answer_type": "compute",
            "required_slots": [
                {"entity": "3m", "period": "fy2022", "metric": "revenue"},
                {"entity": "3m", "period": "fy2022", "metric": "capital expenditure"},
            ],
        },
    )

    slot_anchor_by_metric = {
        str(slot.get("metric", "")): str(slot.get("source_anchor", ""))
        for slot in qs.get("required_slots", [])
        if isinstance(slot, dict)
    }
    assert slot_anchor_by_metric.get("revenue") == "income statement"
    assert slot_anchor_by_metric.get("capital expenditure") == "cash flow statement"


def test_apply_query_state_heuristics_quick_ratio_prefers_direct_metric_slot():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query=(
            "Does 3M have a reasonably healthy liquidity profile based on its quick ratio for "
            "Q2 of FY2023?"
        ),
        query_state={
            "entity": "3M",
            "period": "Q2 of FY2023",
            "metric": "quick ratio",
            "answer_type": "extract",
            "source_anchor": "balance sheet",
            "required_slots": [
                {
                    "entity": "3m",
                    "period": "q2 of fy2023",
                    "metric": "current assets",
                    "source_anchor": "balance sheet",
                },
                {
                    "entity": "3m",
                    "period": "q2 of fy2023",
                    "metric": "current liabilities",
                    "source_anchor": "balance sheet",
                },
            ],
        },
    )
    assert qs.get("metric") == "quick ratio"
    assert qs.get("source_anchor") is None
    slots = qs.get("required_slots", [])
    assert isinstance(slots, list) and len(slots) == 1
    assert isinstance(slots[0], dict)
    assert slots[0].get("metric") == "quick ratio"


def test_apply_query_state_heuristics_capital_intensity_promotes_boolean_and_capex_revenue_slots():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    qs = handler._apply_query_state_heuristics(  # noqa: SLF001
        query="Is CVS Health a capital-intensive business based on FY2022 data?",
        query_state={
            "entity": "CVS Health",
            "period": "FY2022",
            "metric": "capital intensity",
            "answer_type": "compute",
            "required_slots": [
                {
                    "entity": "cvs health",
                    "period": "fy2022",
                    "metric": "total assets",
                    "source_anchor": "balance sheet",
                },
                {
                    "entity": "cvs health",
                    "period": "fy2022",
                    "metric": "total capital expenditures",
                    "source_anchor": "cash flow statement",
                },
            ],
        },
    )
    assert qs.get("answer_type") == "boolean"
    assert qs.get("metric") == "capital intensity"
    slots = qs.get("required_slots", [])
    assert isinstance(slots, list) and len(slots) == 2
    slot_metrics = {str(slot.get("metric", "")) for slot in slots if isinstance(slot, dict)}
    assert "capital expenditures" in slot_metrics
    assert "revenue" in slot_metrics


def test_extract_textual_tool_calls_parses_multiple_calls():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    response = (
        '<tool_call>\n{"name": "graph_search", "arguments": {"entities": ["Scott Derrickson nationality"], "depth": 1}}\n</tool_call>\n'
        '<tool_call>\n{"name": "graph_search", "arguments": {"entities": ["Ed Wood nationality"], "depth": 1}}\n</tool_call>'
    )

    calls = handler._extract_textual_tool_calls(response)  # noqa: SLF001

    assert len(calls) == 2
    assert calls[0]["name"] == "graph_search"
    assert calls[1]["arguments"]["entities"] == ["Ed Wood nationality"]


def test_deterministic_compute_slot_entries_extracts_year_specific_values():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "entity": "Amazon",
        "period": "FY2016,FY2017",
        "metric": "year-over-year change in revenue",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "amazon",
                "period": "fy2016",
                "metric": "revenue",
                "source_anchor": "income statement",
            },
            {
                "entity": "amazon",
                "period": "fy2017",
                "metric": "revenue",
                "source_anchor": "income statement",
            },
        ],
    }
    missing_slots = list(query_state["required_slots"])
    nodes = [
        {
            "title": "AMAZON_2017_10K",
            "page": 43,
            "sent_id": 1001,
            "text": (
                "Consolidated statements of operations. "
                "Year ended December 31, 2016 net sales were 135,987."
            ),
        },
        {
            "title": "AMAZON_2017_10K",
            "page": 43,
            "sent_id": 1002,
            "text": (
                "Consolidated statements of operations. "
                "Year ended December 31, 2017 net sales were 177,866."
            ),
        },
    ]

    entries = handler._deterministic_compute_slot_entries(  # noqa: SLF001
        query_state=query_state,
        missing_slots=missing_slots,
        nodes=nodes,
    )

    assert len(entries) == 2
    values_by_period = {}
    for entry in entries:
        slot_struct = handler._parse_slot_struct(entry.get("slot"))  # noqa: SLF001
        if not slot_struct:
            continue
        values_by_period[slot_struct.get("period")] = str(entry.get("value", ""))
    assert values_by_period.get("fy2016") == "135987"
    assert values_by_period.get("fy2017") == "177866"


def test_deterministic_compute_slot_entries_relaxes_balance_sheet_anchor_for_statement_line_rows():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "entity": "Activision Blizzard",
        "period": "FY2018,FY2019",
        "metric": "fixed asset turnover ratio",
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "activision blizzard",
                "period": "fy2018",
                "metric": "pp&e",
                "source_anchor": "balance sheet",
            },
            {
                "entity": "activision blizzard",
                "period": "fy2019",
                "metric": "pp&e",
                "source_anchor": "balance sheet",
            },
        ],
    }
    nodes = [
        {
            "title": "ACTIVISIONBLIZZARD_2019_10K",
            "page": 90,
            "sent_id": 436,
            "text": "At December 31, 2018, Property and equipment, net was $ 282 million.",
        },
        {
            "title": "ACTIVISIONBLIZZARD_2019_10K",
            "page": 90,
            "sent_id": 437,
            "text": "At December 31, 2019, Property and equipment, net was $ 253 million.",
        },
    ]

    entries = handler._deterministic_compute_slot_entries(  # noqa: SLF001
        query_state=query_state,
        missing_slots=list(query_state["required_slots"]),
        nodes=nodes,
    )

    values_by_period = {}
    for entry in entries:
        slot_struct = handler._parse_slot_struct(entry.get("slot"))  # noqa: SLF001
        if not slot_struct:
            continue
        values_by_period[slot_struct.get("period")] = str(entry.get("value", ""))

    assert values_by_period.get("fy2018") == "282"
    assert values_by_period.get("fy2019") == "253"


def test_collapsed_multi_period_slots_detects_same_value_reused_for_different_year_slots():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "answer_type": "compute",
        "required_slots": [
            {
                "entity": "american express",
                "period": "fy2021",
                "metric": "effective tax rate",
            },
            {
                "entity": "american express",
                "period": "fy2022",
                "metric": "effective tax rate",
            },
        ],
    }
    ledger = [
        {
            "slot": {
                "entity": "american express",
                "period": "fy2021",
                "metric": "effective tax rate",
            },
            "value": "21.6",
            "citation": "[[AMERICANEXPRESS_2022_10K, Page 47, Chunk 224]]",
        },
        {
            "slot": {
                "entity": "american express",
                "period": "fy2022",
                "metric": "effective tax rate",
            },
            "value": "21.6",
            "citation": "[[AMERICANEXPRESS_2022_10K, Page 47, Chunk 224]]",
        },
    ]

    collapsed = handler._collapsed_multi_period_slots(query_state, ledger)  # noqa: SLF001
    assert isinstance(collapsed, list)
    assert len(collapsed) == 2


def test_value_matches_slot_period_rejects_quarter_only_span_for_fy_slot():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    assert handler._value_matches_slot_period(  # noqa: SLF001
        value="436",
        slot_period="FY2023",
        citation="[[AMCOR_2023Q4_EARNINGS, Page 13, Chunk 70]]",
        citation_span="For the three months ended June 30, 2023, adjusted EBITDA was 436 million.",
    ) is False
    assert handler._value_matches_slot_period(  # noqa: SLF001
        value="2,018 million",
        slot_period="FY2023",
        citation="[[AMCOR_2023Q4_EARNINGS, Page 13, Chunk 70]]",
        citation_span="For the twelve months ended June 30, 2023, adjusted non-GAAP EBITDA was 2,018 million.",
    ) is True


def test_should_force_missing_slot_rescue_requires_rich_context():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {
        "answer_type": "compute",
        "required_slots": [{"metric": "revenue"}],
    }
    diagnostics = {
        "accepted_entries": 0,
        "reject_reasons": {"slot_mismatch": 2},
    }

    should_rescue = handler._should_force_missing_slot_rescue(  # noqa: SLF001
        query_state=query_state,
        model_missing_slots=[{"metric": "revenue"}],
        diagnostics=diagnostics,
        context_citation_count=5,
    )
    should_not_rescue = handler._should_force_missing_slot_rescue(  # noqa: SLF001
        query_state=query_state,
        model_missing_slots=[{"metric": "revenue"}],
        diagnostics=diagnostics,
        context_citation_count=0,
    )

    assert should_rescue is True
    assert should_not_rescue is False


def test_should_force_missing_slot_rescue_for_period_mismatch_single_slot():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    should_rescue = handler._should_force_missing_slot_rescue(  # noqa: SLF001
        query_state={"answer_type": "extract", "required_slots": [{"metric": "net pp&e"}]},
        model_missing_slots=[{"metric": "net pp&e"}],
        diagnostics={"accepted_entries": 0, "reject_reasons": {"value_period_mismatch": 1}},
        context_citation_count=1,
    )
    assert should_rescue is True


def test_resolve_required_slot_key_uses_year_hint_from_citation_title():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    slot_2016 = {
        "entity": "amazon",
        "period": "fy2016",
        "metric": "revenue",
        "source_anchor": "income statement",
    }
    slot_2017 = {
        "entity": "amazon",
        "period": "fy2017",
        "metric": "revenue",
        "source_anchor": "income statement",
    }
    required_map = {
        handler._normalize_slot(slot_2016): slot_2016,  # noqa: SLF001
        handler._normalize_slot(slot_2017): slot_2017,  # noqa: SLF001
    }
    ambiguous_slot = {
        "entity": "amazon",
        "period": "",
        "metric": "revenue",
        "source_anchor": "income statement",
    }

    resolved_key, reason = handler._resolve_required_slot_key(  # noqa: SLF001
        slot_raw=ambiguous_slot,
        required_map=required_map,
        query_state={"entity": "Amazon"},
        value="177,866",
        citation="[[AMAZON_2017_10K, Page 49, Chunk 307]]",
        citation_span="Net sales in 2017 were 177,866 and in 2016 were 135,987.",
    )

    assert resolved_key == handler._normalize_slot(slot_2017)  # noqa: SLF001
    assert reason == "structural_slot_tiebreak_year_hint"


def test_build_search_entities_compute_keeps_multi_year_slot_coverage_before_query():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query = (
        "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 "
        "using the statement of income?"
    )
    query_state = {
        "entity": "Amazon",
        "period": "FY2016-FY2017",
        "metric": "revenue",
        "source_anchor": "income statement",
        "answer_type": "compute",
    }
    missing_slots = [
        {
            "entity": "amazon",
            "period": "fy2016",
            "metric": "revenue",
            "source_anchor": "income statement",
        },
        {
            "entity": "amazon",
            "period": "fy2017",
            "metric": "revenue",
            "source_anchor": "income statement",
        },
    ]

    entities = handler._build_search_entities(  # noqa: SLF001
        query=query,
        query_state=query_state,
        missing_slots=missing_slots,
    )

    assert entities
    assert entities[0] != query
    assert "amazon fy2016 revenue income statement" in entities[:8]
    assert "amazon fy2017 revenue income statement" in entities[:12]


def test_filter_nodes_by_query_entity_prefers_period_overlap():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    nodes = [
        {"title": "3M_2015_10K", "page": 1, "sent_id": 10, "text": "old"},
        {"title": "3M_2022_10K", "page": 2, "sent_id": 20, "text": "target"},
    ]
    filtered = handler._filter_nodes_by_query_entity(  # noqa: SLF001
        nodes,
        {
            "entity": "3M",
            "period": "FY2022",
        },
        fail_open=False,
    )
    assert len(filtered) == 1
    assert filtered[0]["title"] == "3M_2022_10K"


def test_filter_nodes_by_query_entity_debt_listing_disables_strict_period_pruning():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    nodes = [
        {"title": "3M_2023Q2_10Q", "page": 1, "sent_id": 1, "text": "cover"},
        {"title": "3M_2022_10K", "page": 1, "sent_id": 2, "text": "section 12(b) table"},
    ]
    filtered = handler._filter_nodes_by_query_entity(  # noqa: SLF001
        nodes,
        {
            "entity": "3M",
            "period": "Q2 2023",
            "metric": "debt securities registered to trade on a national securities exchange",
            "answer_type": "list",
        },
        fail_open=False,
    )
    titles = {str(node.get("title")) for node in filtered}
    assert "3M_2023Q2_10Q" in titles
    assert "3M_2022_10K" in titles


def test_filter_nodes_by_query_entity_period_mismatch_is_strict_for_non_relaxed_path():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    nodes = [
        {"title": "CVSHEALTH_2019_10K", "page": 1, "sent_id": 11, "text": "old period"},
    ]

    strict = handler._filter_nodes_by_query_entity(  # noqa: SLF001
        nodes,
        {
            "entity": "CVS Health",
            "period": "FY2022",
            "metric": "capital intensity",
            "answer_type": "boolean",
        },
        fail_open=False,
    )
    relaxed = handler._filter_nodes_by_query_entity(  # noqa: SLF001
        nodes,
        {
            "entity": "CVS Health",
            "period": "FY2022",
            "metric": "capital intensity",
            "answer_type": "boolean",
        },
        fail_open=True,
    )

    assert strict == []
    assert len(relaxed) == 1


def test_prefer_refined_candidate_allows_insufficient_rollback_when_issues_improve():
    service = AgentService(
        model_id="local",
        strategy="hyporeflect",
        llm_override=MagicMock(),
        grag_override=MagicMock(),
    )
    state = AgentState("Q", [])
    state.query_state = {"missing_data_policy": "insufficient"}

    keep_after, quality = service._orchestrator.refinement_loop._prefer_refined_candidate(  # noqa: SLF001
        state=state,
        before_answer="@@ANSWER: yes [[DOC_A, Page 1, Chunk 1]]",
        before_passed=False,
        before_meta={"issues": ["u1", "u2", "u3"]},
        after_answer="@@ANSWER: insufficient evidence",
        after_passed=False,
        after_meta={"issues": ["u1"]},
    )

    assert keep_after is True
    assert quality.get("reason") == "issue_count_improved_on_insufficient_rollback"


def test_prefer_refined_candidate_keeps_grounded_when_before_passed():
    service = AgentService(
        model_id="local",
        strategy="hyporeflect",
        llm_override=MagicMock(),
        grag_override=MagicMock(),
    )
    state = AgentState("Q", [])
    state.query_state = {"missing_data_policy": "insufficient"}

    keep_after, quality = service._orchestrator.refinement_loop._prefer_refined_candidate(  # noqa: SLF001
        state=state,
        before_answer="@@ANSWER: yes [[DOC_A, Page 1, Chunk 1]]",
        before_passed=True,
        before_meta={"issues": []},
        after_answer="@@ANSWER: insufficient evidence",
        after_passed=False,
        after_meta={"issues": ["u1"]},
    )

    assert keep_after is False
    assert quality.get("reason") == "non_regression_guard_before_grounded_after_insufficient"


def test_prefer_refined_candidate_blocks_inapplicable_policy_insufficient_rollback():
    service = AgentService(
        model_id="local",
        strategy="hyporeflect",
        llm_override=MagicMock(),
        grag_override=MagicMock(),
    )
    state = AgentState("Q", [])
    state.query_state = {"missing_data_policy": "inapplicable_explain"}
    state.missing_slots = [{"metric": "quick ratio"}]
    state.context = "[[DOC_A, Page 1, Chunk 1]]\nQuick ratio section"

    keep_after, quality = service._orchestrator.refinement_loop._prefer_refined_candidate(  # noqa: SLF001
        state=state,
        before_answer="@@ANSWER: no [[DOC_A, Page 1, Chunk 1]]",
        before_passed=False,
        before_meta={"issues": ["u1", "u2"]},
        after_answer="@@ANSWER: insufficient evidence [[DOC_A, Page 1, Chunk 1]]",
        after_passed=False,
        after_meta={"issues": []},
    )

    assert keep_after is False
    assert quality.get("reason") == "policy_disallows_insufficient_after_grounded_answer"


@pytest.mark.asyncio
async def test_run_refinement_loop_restores_reflection_meta_on_rollback():
    service = AgentService(
        model_id="local",
        strategy="hyporeflect",
        llm_override=MagicMock(),
        grag_override=MagicMock(),
    )
    state = AgentState("Q", [])
    state.query_state = {"missing_data_policy": "insufficient"}
    state.final_answer = "@@ANSWER: insufficient evidence"
    state.critique = "FAIL"
    before_meta = {
        "decision": "FAIL",
        "arithmetic_check": "na",
        "issues": ["before_issue"],
        "accepted": True,
    }
    state.reflection_meta = dict(before_meta)

    async def _mock_refinement_run(s):
        s.final_answer = "@@ANSWER: yes [[DOC_A, Page 1, Chunk 1]]"
        s.critique = "FAIL after"

    async def _mock_reflection_run(s):
        s.reflection_meta = {
            "decision": "FAIL",
            "arithmetic_check": "na",
            "issues": ["after_issue_1", "after_issue_2"],
            "accepted": True,
        }
        return False

    service.refinement.run = AsyncMock(side_effect=_mock_refinement_run)
    service.reflection.run = AsyncMock(side_effect=_mock_reflection_run)
    service._orchestrator.refinement_loop._prefer_refined_candidate = MagicMock(return_value=(False, {"reason": "forced_rollback"}))  # noqa: SLF001

    await service._orchestrator.refinement_loop.run_loop(state, reflection_passed=False)  # noqa: SLF001

    assert state.final_answer == "@@ANSWER: insufficient evidence"
    assert state.reflection_meta == before_meta


def test_extract_primary_financial_number_ignores_company_token_number():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    text = "Net sales for 3M Company and Subsidiaries were $34,229 million in 2022."

    value = handler._extract_primary_financial_number(text)  # noqa: SLF001

    assert value == pytest.approx(34229.0)


@pytest.mark.asyncio
async def test_extract_evidence_entries_rejects_value_not_in_cited_span():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={
        "entries": [
            {
                "slot": {
                    "entity": "3m",
                    "period": "2022",
                    "metric": "segment growth",
                },
                "value": "Consumer segment declined by 0.9%",
                "citation": "[[3M_2022_10K, Page 48, Chunk 269]]",
            }
        ],
        "missing_slots": [],
    })
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "2022",
        "metric": "segment growth",
        "answer_type": "extract",
        "required_slots": [
            {"entity": "3m", "period": "2022", "metric": "segment growth"},
        ],
    }
    context = (
        "[[3M_2022_10K, Page 48, Chunk 269]]\n"
        "Net sales were $34,229 million in 2022."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={},
    )

    assert entries == []
    assert missing_slots
    assert diagnostics.get("reject_reasons", {}).get("value_not_in_cited_span", 0) >= 1


@pytest.mark.asyncio
async def test_extract_evidence_entries_allows_capex_cashflow_relaxed_value_span_match():
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={
        "entries": [
            {
                "slot": {
                    "entity": "3m",
                    "period": "fy2018",
                    "metric": "capital expenditure",
                    "source_anchor": "cash flow statement",
                },
                "value": "1577",
                "citation": "[[3M_2018_10K, Page 59, Chunk 365]]",
            }
        ],
        "missing_slots": [],
    })
    handler = ExecutionHandler(llm=llm, grag=MagicMock())
    query_state = {
        "entity": "3M",
        "period": "FY2018",
        "metric": "capital expenditure",
        "answer_type": "extract",
        "source_anchor": "cash flow statement",
        "required_slots": [
            {
                "entity": "3m",
                "period": "fy2018",
                "metric": "capital expenditure",
                "source_anchor": "cash flow statement",
            }
        ],
    }
    context = (
        "[[3M_2018_10K, Page 59, Chunk 365]]\n"
        "Purchases of property, plant and equipment were (1,577) for FY2018."
    )

    entries, missing_slots, diagnostics = await handler._extract_evidence_entries(  # noqa: SLF001
        query_state=query_state,
        context_excerpt=context,
        filter_policy={},
    )

    assert len(entries) == 1
    assert missing_slots == []
    assert diagnostics.get("reject_reasons", {}).get("value_not_in_cited_span", 0) == 0


def test_verify_answer_grounding_rejects_non_compute_citation_outside_context():
    handler = ExecutionHandler(llm=MagicMock(), grag=MagicMock())
    query_state = {"answer_type": "extract", "missing_data_policy": "insufficient"}
    answer = "@@ANSWER: Electronics and Energy [[DOC_B, Page 1, Chunk 1]]"
    context = "[[DOC_A, Page 1, Chunk 1]]\nConsumer segment declined."
    ledger = [{"slot": "segment", "value": "Electronics and Energy", "citation": "[[DOC_B, Page 1, Chunk 1]]"}]

    ok, reason = handler._verify_answer_grounding(  # noqa: SLF001
        answer=answer,
        query_state=query_state,
        evidence_ledger=ledger,
        context=context,
        missing_slots=[],
    )

    assert ok is False
    assert "citation not present in context" in reason


@pytest.mark.asyncio
async def test_tool_call_response_processes_multiple_calls_with_budget():
    handler = ExecutionHandler(llm=AsyncMock(), grag=MagicMock())
    state = AgentState("What is A FY2022 net income and FY2023 net income?", [])
    state.query_state = {"answer_type": "extract"}
    loop_state = ExpansionLoopState(max_tool_calls=2, tool_calls_used=0)

    handler._handle_retrieval_tool_call = AsyncMock()

    resp = MagicMock()
    resp.tool_calls = [
        MagicMock(
            function=MagicMock(name="graph_search", arguments='{"entities":["A"],"depth":1}'),
            id="tc1",
        ),
        MagicMock(
            function=MagicMock(name="graph_search", arguments='{"entities":["B"],"depth":1}'),
            id="tc2",
        ),
    ]

    await handler._handle_tool_call_response(state, turn=0, resp=resp, loop_state=loop_state)

    assert handler._handle_retrieval_tool_call.await_count == 2
    assert loop_state.tool_calls_used == 0
    assert not any(
        isinstance(item, dict) and item.get("step") == "tool_calls_truncated_0"
        for item in state.trace
    )


@pytest.mark.asyncio
async def test_tool_call_response_stops_when_turn_budget_exhausted():
    handler = ExecutionHandler(llm=AsyncMock(), grag=MagicMock())
    state = AgentState("What is A FY2022 net income?", [])
    state.query_state = {"answer_type": "extract"}
    loop_state = ExpansionLoopState(max_tool_calls=1, tool_calls_used=1)

    handler._handle_retrieval_tool_call = AsyncMock()

    resp = MagicMock()
    resp.tool_calls = [
        MagicMock(
            function=MagicMock(name="graph_search", arguments='{"entities":["A"],"depth":1}'),
            id="tc1",
        )
    ]

    await handler._handle_tool_call_response(state, turn=0, resp=resp, loop_state=loop_state)

    assert handler._handle_retrieval_tool_call.await_count == 0
    assert any(
        isinstance(item, dict) and item.get("step") == "tool_calls_truncated_0"
        for item in state.trace
    )


@pytest.mark.asyncio
async def test_tool_call_response_partial_budget_truncates_remaining_calls():
    handler = ExecutionHandler(llm=AsyncMock(), grag=MagicMock())
    state = AgentState("What is A FY2022 net income and FY2023 net income?", [])
    state.query_state = {"answer_type": "extract"}
    loop_state = ExpansionLoopState(max_tool_calls=2, tool_calls_used=0)

    handler._handle_retrieval_tool_call = AsyncMock()

    resp = MagicMock()
    resp.tool_calls = [
        MagicMock(
            function=MagicMock(name="graph_search", arguments='{"entities":["A"],"depth":1}'),
            id="tc1",
        ),
        MagicMock(
            function=MagicMock(name="graph_search", arguments='{"entities":["B"],"depth":1}'),
            id="tc2",
        ),
        MagicMock(
            function=MagicMock(name="graph_search", arguments='{"entities":["C"],"depth":1}'),
            id="tc3",
        ),
    ]

    await handler._handle_tool_call_response(state, turn=0, resp=resp, loop_state=loop_state)

    assert handler._handle_retrieval_tool_call.await_count == 2
    assert any(
        isinstance(item, dict) and item.get("step") == "tool_calls_truncated_0"
        for item in state.trace
    )
