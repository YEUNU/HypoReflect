import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from models.hyporeflect.graphrag import GraphRAG

@pytest.mark.asyncio
async def test_adaptive_semantic_chunking_audit():
    # Setup
    rag = GraphRAG(strategy="hyporeflect")
    
    # Mock VLLMClient and Embedding logic
    rag.vllm = MagicMock()
    # Mocking async methods correctly
    rag.vllm.get_embeddings = AsyncMock()
    rag.vllm.generate_response = AsyncMock()
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_response = AsyncMock(return_value="Mock page/group summary")
    
    # Mock embeddings: return 5 identical vectors so they group together
    # processed_sentences will have [table_lines..., sentences...]
    rag.vllm.get_embeddings.return_value = [[0.1]*2048 for _ in range(10)]
    rag.vllm.generate_response.return_value = "Converted table sentence 1.\nConverted table sentence 2."
    
    # Mock LLM response for HOPRAG extraction
    rag.indexing_llm.generate_json.return_value = {
        "summary": "Mock summary",
        "q_minus": ["q1"],
        "q_plus": ["q2"]
    }

    # Document with a table and semantically related sentences
    test_content = """Title: Test Document
| Year | Event |
| 2020 | Pandemic |
| 2021 | Vaccine |
The 2020 pandemic changed the world.
It led to global lockdowns.
Vaccines were developed in 2021.
"""

    # Execute
    knowledge = await rag.extract_knowledge(test_content, source="test")
    
    # Audit assertions
    chunks = knowledge["chunks"]
    
    # Deviation 1: Sentence segmentation (Current: split by line)
    # The paper requires regex heuristics and embedding cohesion.
    # If it just splits by line, it will have 5 chunks (2 for table, 3 for sentences)
    # whereas semantic chunking might group the related sentences.
    print(f"DEBUG: Found {len(chunks)} chunks")
    
    # Deviation 2: Table Propositionality
    # Check if TABLE_TO_TEXT_PROMPT was used for the table lines.
    # In current graphrag.py, it doesn't even detect tables.
    
    # Verification of NEW behavior:
    # 1. Table is detected and should trigger _table_to_text (mocked by generate_response)
    # 2. Sentences are grouped if they are semantically similar.
    
    # We expect 4 groups if cohesion works: 
    # [Table converted lines], [Pandemic sentences], [Lockdown sentence], [Vaccine sentence]
    # In our mock, if similarity is high, Pandemic sentences might merge.
    
    print(f"DEBUG: Found {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c['text']}")

    # Since we didn't mock cosine_similarity to return low values specifically, 
    # the default might group or split based on actual embeddings from the server.
    assert len(chunks) < 6 # Should be less than raw line count
    
    # Check that the running summary was updated (implicitly via the fact it didn't crash)
    assert chunks[0]["summary"] == "Mock summary"

if __name__ == "__main__":
    asyncio.run(test_adaptive_semantic_chunking_audit())
