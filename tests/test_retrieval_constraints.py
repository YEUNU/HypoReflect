import asyncio

import pytest
from unittest.mock import AsyncMock

from core.config import RAGConfig
from models.hyporeflect.graphrag import GraphRAG


def test_extract_query_metadata_captures_company_and_year():
    rag = GraphRAG(strategy="hyporeflect")
    meta = rag._extract_query_metadata(  # noqa: SLF001 - unit test for internal helper
        "Among operations, investing, and financing activities, which brought in the most cash flow for AMD in FY22?"
    )

    assert "amd" in (meta.get("company_keys") or set())
    assert meta.get("financial_intent") is True


def test_q_plus_quality_gate_requires_entity_period_metric_and_anchor():
    rag = GraphRAG(strategy="hyporeflect")
    ok = rag._is_high_quality_q_plus(  # noqa: SLF001 - unit test for internal helper
        "For AMD FY2022 cash flow statement, what was operating cash flow?",
        title="AMD_2022_10K",
        chunk_text="Consolidated Statements of Cash Flows",
    )
    bad_missing_anchor = rag._is_high_quality_q_plus(  # noqa: SLF001 - unit test for internal helper
        "For AMD FY2022 what was operating cash flow?",
        title="AMD_2022_10K",
        chunk_text="Narrative risk factors section only.",
    )
    bad_missing_entity = rag._is_high_quality_q_plus(  # noqa: SLF001 - unit test for internal helper
        "For FY2022 cash flow statement, what was operating cash flow?",
        title="AMD_2022_10K",
        chunk_text="Consolidated Statements of Cash Flows",
    )

    assert ok is True
    assert bad_missing_anchor is False
    assert bad_missing_entity is False


@pytest.mark.asyncio
async def test_retrieve_prefers_company_matched_candidate(monkeypatch):
    rag = GraphRAG(strategy="hyporeflect")

    # Two candidates with identical rerank score; AMD document should win by metadata calibration.
    candidates = [
        {
            "id": "1",
            "title": "AMCOR_2022_10K",
            "sent_id": 10,
            "page": 38,
            "text": "Net cash provided by operating activities was ...",
            "rrf_score": 1.0,
        },
        {
            "id": "2",
            "title": "AMD_2022_10K",
            "sent_id": 267,
            "page": 52,
            "text": "Net cash provided by operating activities was $3.6 billion.",
            "rrf_score": 0.99,
        },
    ]

    rag._hybrid_rrf_candidates = AsyncMock(return_value=candidates)  # type: ignore[method-assign]
    rag.llm.rerank = AsyncMock(return_value=[0.8, 0.8])

    monkeypatch.setattr(RAGConfig, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(RAGConfig, "RERANKER_THRESHOLD", 0.0)

    _, nodes = await rag.retrieve(
        "Among operations, investing, and financing activities, which brought in the most cash flow for AMD in FY22?",
        top_k=1,
    )

    assert nodes
    assert "AMD_2022_10K" in nodes[0]["title"]


@pytest.mark.asyncio
async def test_retrieve_expands_with_q_plus_when_stage1_is_insufficient(monkeypatch):
    rag = GraphRAG(strategy="hyporeflect")
    monkeypatch.setattr(RAGConfig, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(RAGConfig, "RERANKER_THRESHOLD", 0.0)

    stage1_node = {
        "id": "n1",
        "title": "AMD_2022_10K",
        "sent_id": 1,
        "page": 10,
        "text": "Operating cash flow was 3.6 billion.",
        "rrf_score": 1.0,
    }
    stage2_node = {
        "id": "n2",
        "title": "AMD_2022_10K",
        "sent_id": 2,
        "page": 11,
        "text": "Capital expenditures were 1.1 billion in FY2022.",
        "rrf_score": 0.9,
    }

    async def fake_candidates(q_text: str, limit: int, channel: str = "body"):
        if channel == "q_minus":
            return [dict(stage1_node)]
        if channel == "q_plus":
            return [dict(stage2_node)]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    rag.llm.rerank = AsyncMock(side_effect=lambda query, docs, instruction=None: [0.7 for _ in docs])

    _, nodes = await rag.retrieve(
        "What was AMD FY2022 free cash flow?",
        top_k=2,
    )

    ids = {n.get("id") for n in nodes}
    assert "n1" in ids
    assert "n2" in ids


@pytest.mark.asyncio
async def test_build_graph_filters_q_plus_by_quality_gate():
    rag = GraphRAG(strategy="hyporeflect")
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts: [[0.1] for _ in texts])

    knowledge = {
        "chunks": [
            {
                "text": "Consolidated Statements of Cash Flows for fiscal year 2022.",
                "title": "AMD_2022_10K",
                "sent_id": 0,
                "page": 1,
                "q_minus": ["For AMD FY2022, what was operating cash flow?"],
                "q_plus": [
                    "For AMD FY2022 what was operating cash flow?",
                    "For AMD FY2022 cash flow statement, what was operating cash flow?",
                ],
                "summary": "Cash flow statement summary.",
            }
        ]
    }

    await rag.build_graph(knowledge, source="unit_test", document_filename="AMD_2022_10K.txt")

    assert rag._pending_batch, "Expected build_graph to enqueue at least one batch item."
    payload = rag._pending_batch[-1]["data"][0]
    assert payload["q_plus"] == ["For AMD FY2022 cash flow statement, what was operating cash flow?"]
    assert payload["q_plus_text"] == "For AMD FY2022 cash flow statement, what was operating cash flow?"
