"""
[HopRAG] adapter wired to the official HopRetriever implementation.

This keeps the benchmark interface while delegating traversal logic to
`third_party/HopRAG/HopRetriever.py` and preserving the current generation and
rerank models used in this repository.
"""

import asyncio
import importlib
import logging
import os
from pathlib import Path
import sys
import threading
import types
from typing import Any, Dict, List, Optional, Tuple

from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from utils.formatters import format_context_from_nodes

logger = logging.getLogger(__name__)


def _run_coro_sync(coro):
    """Run async coroutines from synchronous official HopRAG hooks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: Dict[str, Any] = {}
    errors: Dict[str, BaseException] = {}

    def _runner():
        try:
            holder["value"] = asyncio.run(coro)
        except BaseException as e:  # pragma: no cover - defensive fallback
            errors["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in errors:
        raise errors["error"]
    return holder.get("value")


def _install_missing_hoprag_stubs() -> None:
    """Install tiny import-time stubs for optional upstream dependencies."""
    if "paddlenlp" not in sys.modules:
        paddlenlp = types.ModuleType("paddlenlp")

        def _taskflow(*_args, **_kwargs):
            def _run(_text):
                return []

            return _run

        paddlenlp.Taskflow = _taskflow  # type: ignore[attr-defined]
        sys.modules["paddlenlp"] = paddlenlp

    if "sentence_transformers" not in sys.modules:
        sentence_transformers = types.ModuleType("sentence_transformers")

        class _SentenceTransformer:
            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, documents, **_kwargs):
                if isinstance(documents, str):
                    return [0.0] * 768
                return [[0.0] * 768 for _ in documents]

        sentence_transformers.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
        sys.modules["sentence_transformers"] = sentence_transformers

    if "modelscope" not in sys.modules:
        modelscope = types.ModuleType("modelscope")

        class _DummyModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def eval(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

            @property
            def device(self):
                return "cpu"

            def __call__(self, *_args, **_kwargs):  # pragma: no cover - fallback
                return types.SimpleNamespace(logits=[0.0])

        class _DummyTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def __call__(self, *_args, **_kwargs):
                return {}

            def apply_chat_template(self, *_args, **_kwargs):
                return ""

            def batch_decode(self, *_args, **_kwargs):
                return [""]

        modelscope.AutoModelForCausalLM = _DummyModel  # type: ignore[attr-defined]
        modelscope.AutoTokenizer = _DummyTokenizer  # type: ignore[attr-defined]
        modelscope.AutoModelForSequenceClassification = _DummyModel  # type: ignore[attr-defined]
        sys.modules["modelscope"] = modelscope


class HopRAGAdapter:
    """
    HopRAG benchmark adapter that executes the official HopRetriever traversal.
    """

    def __init__(
        self,
        model_id: str = "local",
        max_hop: int = 4,
        top_k: int = 10,
        corpus_tag: str = "default",
    ):
        self.model_id = model_id
        self.max_hop = max_hop
        self.top_k = top_k
        self.corpus_tag = corpus_tag

        self.llm = get_llm_client(model_id)
        self.vllm = VLLMClient(model_name=model_id)
        self.neo4j = Neo4jService()

        self.prefix = "HO_"
        import re as _re
        _safe_corpus = _re.sub(r"[^A-Za-z0-9_]", "_", self.corpus_tag)
        self.chunk_label = f"{self.prefix}{_safe_corpus}_Chunk"
        self.vector_index = f"hoprag_{_safe_corpus}_vector_idx"

        self._hop_module = self._load_official_hop_module()
        self._configure_official_hop_runtime()
        self._retriever = self._build_official_retriever()
        self._agentic_runner = None
        self._agentic_full_service = None

    def _load_official_hop_module(self):
        _install_missing_hoprag_stubs()

        hop_root = Path(__file__).resolve().parents[2] / "third_party" / "HopRAG"
        if not hop_root.exists():
            raise RuntimeError(f"Official HopRAG not found: {hop_root}")

        root_text = str(hop_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        return importlib.import_module("HopRetriever")

    def _configure_official_hop_runtime(self) -> None:
        """Patch official runtime hooks to use this project's infra/services."""
        hop_module = self._hop_module
        tool_module = importlib.import_module("tool")

        neo4j_url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "1q2w3e4r")
        neo4j_dbname = os.environ.get("NEO4J_DB", "neo4j")

        retrieve_node_dense_query = f"""
CALL db.index.vector.queryNodes({{index}}, 80, {{embedding}})
YIELD node AS dense_node, score AS dense_score
RETURN {{
    text: dense_node.text,
    embed: dense_node.embedding,
    keywords: [w IN split(toLower(replace(coalesce(dense_node.chunk_summary, dense_node.text), '\\n', ' ')), ' ') WHERE size(w) > 0],
    id: dense_node.id,
    title: dense_node.title,
    sent_id: coalesce(dense_node.sent_id, 0),
    page: coalesce(dense_node.page, 0),
    source: dense_node.source
}} AS dense_node, dense_score
ORDER BY dense_score DESC
LIMIT 80
"""
        expand_logic_query = f"""
MATCH (dense_node:{self.chunk_label})-[r:NEXT|HOP]-(logic_node:{self.chunk_label})
WHERE dense_node.text=$text
RETURN {{
    text: logic_node.text,
    embed: logic_node.embedding,
    keywords: [w IN split(toLower(replace(coalesce(logic_node.chunk_summary, logic_node.text), '\\n', ' ')), ' ') WHERE size(w) > 0],
    id: logic_node.id,
    title: logic_node.title,
    sent_id: coalesce(logic_node.sent_id, 0),
    page: coalesce(logic_node.page, 0),
    source: logic_node.source
}} AS logic_node
LIMIT 40
"""

        def _load_embed_model(_name):
            # Embeddings are served by the project's vLLM embedding endpoint.
            return object()

        def _get_doc_embeds(documents, _model):
            if isinstance(documents, str):
                emb = _run_coro_sync(self.vllm.get_embedding(documents))
                return emb or []
            docs = [str(d) for d in documents]
            embs = _run_coro_sync(self.vllm.get_embeddings(docs))
            return embs or [[] for _ in docs]

        def _load_language_model(model_name):
            # Use model id as opaque identifier; chat completion is patched below.
            return model_name

        def _get_chat_completion(chat, return_json=True, model=None, max_tokens=4096, keys=None):
            messages = chat if isinstance(chat, list) else [{"role": "user", "content": str(chat)}]
            _ = model, max_tokens  # keep official signature compatibility
            if not return_json:
                response = _run_coro_sync(self.llm.generate_response(messages, temperature=0.0))
                return response, messages

            generated = _run_coro_sync(self.llm.generate_json(messages, temperature=0.0))
            payload = generated if isinstance(generated, dict) else {}

            values = [payload.get(k, "") for k in (keys or [])]
            return (*values, messages)

        patch_targets = [hop_module, tool_module]
        for target in patch_targets:
            target.neo4j_url = neo4j_url
            target.neo4j_user = neo4j_user
            target.neo4j_password = neo4j_password
            target.neo4j_dbname = neo4j_dbname
            target.node_dense_index_name = self.vector_index
            target.retrieve_node_dense_query = retrieve_node_dense_query
            target.expand_logic_query = expand_logic_query
            target.load_embed_model = _load_embed_model
            target.get_doc_embeds = _get_doc_embeds
            target.load_language_model = _load_language_model
            target.get_chat_completion = _get_chat_completion

    def _build_official_retriever(self):
        hop_cls = getattr(self._hop_module, "HopRetriever")
        return hop_cls(
            llm=self.model_id,
            max_hop=self.max_hop,
            entry_type="node",
            if_hybrid=False,
            if_trim=False,
            tol=2,
            topk=max(self.top_k * 2, self.top_k),
            traversal="bfs_node",
            mock_dense=False,
            mock_sparse=False,
            reranker=None,
        )

    async def _run_official_retrieval(self, query: str) -> List[str]:
        try:
            context_texts, _ = await asyncio.to_thread(self._retriever.search_docs, query)
            if not isinstance(context_texts, list):
                return []
            return [str(t) for t in context_texts if isinstance(t, str) and t.strip()]
        except Exception as e:
            logger.warning("Official HopRetriever traversal failed, fallback enabled: %s", e)
            return []

    async def _lookup_nodes_by_text(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        query = f"""
            UNWIND range(0, size($texts) - 1) AS idx
            WITH idx, $texts[idx] AS target_text
            MATCH (n:{self.chunk_label})
            WHERE n.text = target_text
            RETURN idx, n.id AS id, n.title AS title, n.sent_id AS sent_id,
                   n.page AS page, n.text AS text, n.embedding AS embedding
            ORDER BY idx ASC
        """
        async with self.neo4j.driver.session() as session:
            result = await session.run(query, {"texts": texts})  # type: ignore
            rows = [dict(r) async for r in result]

        by_idx: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            idx = int(row.get("idx", -1))
            if idx < 0 or idx in by_idx:
                continue
            by_idx[idx] = row

        ordered: List[Dict[str, Any]] = []
        for idx in range(len(texts)):
            if idx in by_idx:
                node = by_idx[idx]
                node.pop("idx", None)
                ordered.append(node)
        return ordered

    async def _vector_fallback(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_embed = await self.vllm.get_embedding(query)
        if not query_embed:
            return []
        query_cypher = f"""
            CALL db.index.vector.queryNodes('{self.vector_index}', $k, $embedding)
            YIELD node, score
            RETURN node.id as id, node.title as title, node.sent_id as sent_id,
                   node.page as page, node.text as text, node.embedding as embedding, score
        """
        async with self.neo4j.driver.session() as session:
            result = await session.run(  # type: ignore
                query_cypher,
                {"k": max(top_k * 2, 10), "embedding": query_embed},
            )
            return [dict(rec) async for rec in result]

    async def retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        context_texts = await self._run_official_retrieval(query)
        candidates = await self._lookup_nodes_by_text(context_texts)
        if not candidates:
            candidates = await self._vector_fallback(query, top_k=top_k)
        if not candidates:
            return "", []

        texts = [str(n.get("text", "")) for n in candidates]
        rerank_scores = await self.vllm.rerank(query, texts)
        for idx, score in enumerate(rerank_scores):
            if idx < len(candidates):
                candidates[idx]["rerank_score"] = score
        candidates = sorted(
            candidates,
            key=lambda x: float(x.get("rerank_score", x.get("score", 0.0))),
            reverse=True,
        )
        nodes = candidates[:top_k]
        context = format_context_from_nodes(nodes)
        return context, nodes

    async def run_workflow(self, query: str, history: Optional[List[Dict]] = None) -> Tuple[str, List, List]:
        agentic_mode = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
        if agentic_mode == "on":
            agentic_pipeline = str(os.environ.get("RAG_AGENTIC_PIPELINE", "full") or "full").strip().lower()
            if agentic_pipeline == "lite":
                if self._agentic_runner is None:
                    from models.agentic_core import AgenticOrchestrator

                    self._agentic_runner = AgenticOrchestrator(
                        llm=self.llm,
                        backend=self,
                        strategy_name="hoprag",
                        top_k=self.top_k,
                    )
                return await self._agentic_runner.run(query, history)

            if self._agentic_full_service is None:
                from models.agentic_core import RetrievalGraphAdapter
                from models.hyporeflect.service import AgentService

                graph_adapter = RetrievalGraphAdapter(
                    backend=self,
                    strategy_name="hoprag",
                    default_top_k=self.top_k,
                )
                self._agentic_full_service = AgentService(
                    strategy="hyporeflect",
                    corpus_tag=self.corpus_tag,
                    llm_override=self.llm,
                    grag_override=graph_adapter,
                )
            return await self._agentic_full_service.run_workflow(query, history)

        _ = history
        context, nodes = await self.retrieve(query, top_k=self.top_k)
        if not context:
            return "Unable to find relevant information.", [], []

        prompt = f"""You are a helpful assistant. Answer the question using only the provided context.
If the context is insufficient, say you do not know.

Context:
{context}

Question: {query}

Answer:"""
        messages = [{"role": "user", "content": prompt}]
        answer = await self.llm.generate_response(messages)
        trace = [{"step": "hoprag_official_hopretriever_qa", "input": messages, "output": answer}]
        sources = [
            {
                "doc": n.get("title", ""),
                "page": n.get("page", 0),
                "text": n.get("text", ""),
                "sent_id": n.get("sent_id", 0),
            }
            for n in nodes
        ]
        return answer, sources, trace
