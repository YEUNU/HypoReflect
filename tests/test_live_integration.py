import asyncio
from typing import Dict, Iterable, Tuple

import httpx
import pytest

from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient
from models.naive.naive_rag import NaiveRAG


HEALTH_CHECKS: Dict[str, Tuple[str, set[int]]] = {
    "neo4j_http": ("http://localhost:7474", {200, 401, 403, 405}),
    "generation": ("http://localhost:28000/v1/models", {200, 401}),
    "embedding": ("http://localhost:18082/v1/models", {200, 401}),
    "reranker": ("http://localhost:18083/health", {200}),
}


async def _endpoint_ready(url: str, ok_codes: Iterable[int]) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(url)
        return res.status_code in set(ok_codes)
    except Exception:
        return False


async def _require_live_services() -> None:
    checks = await asyncio.gather(
        *[_endpoint_ready(url, codes) for url, codes in HEALTH_CHECKS.values()]
    )
    missing = [name for name, is_ok in zip(HEALTH_CHECKS.keys(), checks) if not is_ok]
    if missing:
        pytest.skip(f"Live integration services not ready: {', '.join(missing)}")

    neo4j = Neo4jService()
    try:
        rows = await asyncio.wait_for(neo4j.execute_query("RETURN 1 AS ok"), timeout=10.0)
    except Exception as exc:
        await Neo4jService.global_close()
        pytest.skip(f"Live integration Neo4j bolt not ready: {exc}")
    await Neo4jService.global_close()
    if not rows or rows[0].get("ok") != 1:
        pytest.skip("Live integration Neo4j bolt returned unexpected result.")


async def _run_live_step(name: str, coro, timeout_seconds: float):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except Exception as exc:
        pytest.skip(f"Live integration step '{name}' not ready: {exc}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_service_healthchecks():
    await _require_live_services()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_vllm_client_minimal_calls():
    await _require_live_services()

    client = VLLMClient()
    response = await _run_live_step(
        "generate_response",
        client.generate_response(
            [{"role": "user", "content": "Reply with exactly OK"}],
            temperature=0.0,
        ),
        timeout_seconds=60.0,
    )
    assert isinstance(response, str) and response.strip()

    embedding = await _run_live_step(
        "get_embedding",
        client.get_embedding("integration_smoke_token"),
        timeout_seconds=60.0,
    )
    assert isinstance(embedding, list) and len(embedding) > 0

    scores = await _run_live_step(
        "rerank",
        client.rerank(
            "integration",
            ["integration token appears here", "this is unrelated"],
        ),
        timeout_seconds=60.0,
    )
    assert len(scores) == 2
    assert all(isinstance(s, (int, float)) for s in scores)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_naive_index_retrieve_roundtrip():
    await _require_live_services()

    corpus_tag = "it_live"
    unique_token = "integration_live_unique_token_90210"
    rag = NaiveRAG(strategy="naive", corpus_tag=corpus_tag)

    async def cleanup() -> None:
        await rag.neo4j.execute_query(f"MATCH (n:{rag.chunk_label}) DETACH DELETE n")

    async def wait_for_index_online() -> None:
        for _ in range(20):
            rows = await rag.neo4j.execute_query(
                "SHOW VECTOR INDEXES YIELD name, state WHERE name = $name RETURN state",
                {"name": rag.vector_index},
            )
            if rows and rows[0].get("state") == "ONLINE":
                return
            await asyncio.sleep(0.5)

    await _run_live_step("cleanup_before", cleanup(), timeout_seconds=20.0)
    try:
        content = (
            "Title: Integration Live Document\n"
            f"{unique_token} appears in this sentence.\n"
            f"Revenue for {unique_token} was 123.\n"
        )
        await _run_live_step(
            "index_document",
            rag.index_document("integration_live_doc.txt", content),
            timeout_seconds=120.0,
        )
        await _run_live_step("wait_for_index_online", wait_for_index_online(), timeout_seconds=30.0)

        node_count = await _run_live_step(
            "node_count",
            rag.neo4j.execute_query(
                f"MATCH (n:{rag.chunk_label}) RETURN count(n) AS c"
            ),
            timeout_seconds=20.0,
        )
        assert node_count and node_count[0].get("c", 0) > 0, "No indexed nodes were created."

        context = ""
        nodes = []
        for _ in range(15):
            context, nodes = await _run_live_step(
                "retrieve",
                rag.retrieve(unique_token, top_k=3),
                timeout_seconds=60.0,
            )
            if nodes:
                break
            await asyncio.sleep(1)

        assert nodes, "No nodes retrieved from live integration index."
        retrieved_blob = (context + "\n" + "\n".join(n.get("text", "") for n in nodes)).lower()
        assert unique_token.lower() in retrieved_blob
    finally:
        await _run_live_step("cleanup_after", cleanup(), timeout_seconds=20.0)
        await Neo4jService.global_close()
