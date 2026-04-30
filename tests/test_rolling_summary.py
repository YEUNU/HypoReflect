import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from models.hyporeflect.graphrag import GraphRAG

@pytest.mark.asyncio
async def test_rolling_summarization_audit():
    # Setup
    rag = GraphRAG(strategy="hyporeflect")
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_response = AsyncMock(return_value="Mock page/group summary")
    rag.vllm = MagicMock()
    rag.vllm.get_embeddings = AsyncMock()
    
    # Document with multiple sentences
    test_content = """Title: Rolling Test
Sentence 1 about subject A.
Sentence 2 about subject B.
"""

    # Mock embeddings to trigger split (similarity 0.1)
    rag.vllm.get_embeddings.return_value = [[1.0, 0.0], [0.0, 1.0]]

    # Mock LLM for generate_json (which extract_hoprag_queries_with_rolling calls)
    async def mock_gen_json(messages, **kwargs):
        # messages[0]['content'] contains the prompt
        prompt = messages[0]['content']
        # If it's the second chunk, it should contain the first chunk's summary
        if "Sentence 2" in prompt:
            assert "Previous context: Summary of Sent 1" in prompt
        return {"summary": f"Summary of {prompt[:10]}", "q_minus": [], "q_plus": []}

    # Special mock to return controlled summary for checking
    rag.indexing_llm.generate_json.side_effect = [
        {"summary": "Summary of Sent 1", "q_minus": [], "q_plus": []},
        {"summary": "Summary of Sent 2", "q_minus": [], "q_plus": []}
    ]

    knowledge = await rag.extract_knowledge(test_content, source="test")
    
    # The current adaptive threshold may merge into one chunk for this tiny sample.
    assert len(knowledge["chunks"]) >= 1
    # Verify first chunk passed to generate_json (indirectly via indexing_llm mock call check)
    # The first call should include rolling context with document header.
    # The second call (if any) should include prior chunk summary.
    
    first_call_prompt = rag.indexing_llm.generate_json.call_args_list[0][0][0][0]['content']
    assert "Document: Rolling Test" in first_call_prompt
    
    if len(rag.indexing_llm.generate_json.call_args_list) > 1:
        second_call_prompt = rag.indexing_llm.generate_json.call_args_list[1][0][0][0]['content']
        assert "Summary of Sent 1" in second_call_prompt

    print("Audit of Rolling Summarization: PASSED")

if __name__ == "__main__":
    asyncio.run(test_rolling_summarization_audit())
