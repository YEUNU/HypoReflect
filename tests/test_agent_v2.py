import pytest
from unittest.mock import AsyncMock, MagicMock

from models.hyporeflect.service import AgentService


@pytest.mark.asyncio
async def test_agent_5_stage_loop_and_strict_reflection():
    service = AgentService(
        model_id="local",
        strategy="hyporeflect",
        llm_override=MagicMock(),
        grag_override=MagicMock(),
    )

    async def _mock_execution_run(state):
        state.query_state = {
            "answer_type": "extract",
            "metric": "ceo",
            "missing_data_policy": "insufficient",
        }
        state.final_answer = "@@ANSWER: The CEO of A is John [[A, Page 1, Chunk 0]]."
        state.missing_slots = []
        state.evidence_ledger = [
            {
                "slot": "ceo",
                "value": "John",
                "citation": "[[A, Page 1, Chunk 0]]",
            }
        ]
        state.context = "[[A, Page 1, Chunk 0]] John is the CEO."
        state.all_context_data = [
            {"title": "A", "page": 1, "sent_id": 0, "text": "John is the CEO."}
        ]

    service.perception.run = AsyncMock(return_value=None)
    service.planning.run = AsyncMock(return_value=None)
    service.execution.run = AsyncMock(side_effect=_mock_execution_run)
    service.reflection.run = AsyncMock(return_value=True)
    service._orchestrator.refinement_loop.run_loop = AsyncMock(return_value=True)  # noqa: SLF001


    answer, sources, trace = await service.run_workflow("Who is the CEO of A?")

    assert "@@ANSWER:" in answer
    assert "[[A, Page 1, Chunk 0]]" in answer
    assert len(sources) == 1
    assert sources[0]["doc"] == "A"
    assert sources[0]["page"] == 1
    assert sources[0]["sent_id"] == 0
    assert isinstance(trace, list)
    service.perception.run.assert_awaited_once()
    service.planning.run.assert_awaited_once()
    service.execution.run.assert_awaited_once()
    service.reflection.run.assert_awaited_once()
    service._orchestrator.refinement_loop.run_loop.assert_awaited_once()  # noqa: SLF001


@pytest.mark.asyncio
async def test_agent_strict_reflection_failure_and_refinement():
    service = AgentService(
        model_id="local",
        strategy="hyporeflect",
        llm_override=MagicMock(),
        grag_override=MagicMock(),
    )

    async def _mock_execution_run(state):
        state.query_state = {
            "answer_type": "extract",
            "metric": "ceo",
            "missing_data_policy": "insufficient",
        }
        # Initial answer intentionally weak to force refinement path expectations.
        state.final_answer = "@@ANSWER: John is the CEO."
        state.missing_slots = []
        state.evidence_ledger = []
        state.context = "[[A, Page 1, Chunk 0]] John is the CEO."
        state.all_context_data = [
            {"title": "A", "page": 1, "sent_id": 0, "text": "John is the CEO."}
        ]

    async def _mock_refinement_loop(state, reflection_passed):
        assert reflection_passed is False
        state.final_answer = "@@ANSWER: John is the CEO [[A, Page 1, Chunk 0]]."
        return True

    service.perception.run = AsyncMock(return_value=None)
    service.planning.run = AsyncMock(return_value=None)
    service.execution.run = AsyncMock(side_effect=_mock_execution_run)
    service.reflection.run = AsyncMock(return_value=False)
    service._orchestrator.refinement_loop.run_loop = AsyncMock(side_effect=_mock_refinement_loop)  # noqa: SLF001


    answer, _, _ = await service.run_workflow("Who is the CEO?")

    assert "@@ANSWER:" in answer
    assert "[[A, Page 1, Chunk 0]]" in answer
    service.reflection.run.assert_awaited_once()
    service._orchestrator.refinement_loop.run_loop.assert_awaited_once()  # noqa: SLF001
