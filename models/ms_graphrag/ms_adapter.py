"""
[MS GraphRAG] adapter wired to official GlobalSearch/LocalSearch classes.

This keeps benchmark compatibility while using the upstream Microsoft query
orchestration code and preserving current generation/rerank model services.
"""

import importlib
import json
import logging
import os
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client

logger = logging.getLogger(__name__)


def _install_ms_optional_stubs() -> None:
    """Install tiny import-time stubs for optional upstream dependencies."""
    if "nest_asyncio2" not in sys.modules:
        nest_asyncio2 = types.ModuleType("nest_asyncio2")
        nest_asyncio2.apply = lambda: None  # type: ignore[attr-defined]
        sys.modules["nest_asyncio2"] = nest_asyncio2

    if "pandas" not in sys.modules:
        pandas = types.ModuleType("pandas")

        class _DataFrame(list):
            def __init__(self, data=None, *args, **kwargs):
                _ = args, kwargs
                super().__init__(data if isinstance(data, list) else [])

            @property
            def columns(self):
                return []

        pandas.DataFrame = _DataFrame  # type: ignore[attr-defined]
        sys.modules["pandas"] = pandas

    if "litellm" not in sys.modules:
        litellm = types.ModuleType("litellm")
        litellm.AnthropicThinkingParam = dict  # type: ignore[attr-defined]
        litellm.ChatCompletionAudioParam = dict  # type: ignore[attr-defined]
        litellm.ChatCompletionModality = str  # type: ignore[attr-defined]
        litellm.ChatCompletionPredictionContentParam = dict  # type: ignore[attr-defined]
        litellm.OpenAIWebSearchOptions = dict  # type: ignore[attr-defined]
        litellm.ModelResponse = object  # type: ignore[attr-defined]
        litellm.model_cost = {}  # type: ignore[attr-defined]
        litellm.supports_response_schema = lambda *_a, **_kw: False  # type: ignore[attr-defined]
        litellm.encode = lambda _model, text: [ord(ch) for ch in str(text)]  # type: ignore[attr-defined]
        litellm.decode = lambda _model, tokens: "".join(chr(int(t)) for t in tokens)  # type: ignore[attr-defined]

        litellm_exceptions = types.ModuleType("litellm.exceptions")
        for exc_name in [
            "APIError",
            "AuthenticationError",
            "BadRequestError",
            "RateLimitError",
            "ServiceUnavailableError",
            "Timeout",
        ]:
            setattr(litellm_exceptions, exc_name, type(exc_name, (Exception,), {}))

        litellm.exceptions = litellm_exceptions  # type: ignore[attr-defined]
        sys.modules["litellm"] = litellm
        sys.modules["litellm.exceptions"] = litellm_exceptions

    if "json_repair" not in sys.modules:
        json_repair = types.ModuleType("json_repair")
        json_repair.repair_json = lambda text, **_kwargs: text  # type: ignore[attr-defined]
        sys.modules["json_repair"] = json_repair


def _append_ms_package_paths() -> None:
    root = Path(__file__).resolve().parents[2] / "third_party" / "MS_GraphRAG" / "packages"
    package_dirs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    for p in package_dirs:
        if p.exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)


def _load_official_ms_components() -> Dict[str, Any]:
    _install_ms_optional_stubs()
    _append_ms_package_paths()

    global_search_module = importlib.import_module(
        "graphrag.query.structured_search.global_search.search"
    )
    local_search_module = importlib.import_module(
        "graphrag.query.structured_search.local_search.search"
    )
    context_module = importlib.import_module("graphrag.query.context_builder.builders")
    reduce_prompt_module = importlib.import_module(
        "graphrag.prompts.query.global_search_reduce_system_prompt"
    )
    return {
        "GlobalSearch": getattr(global_search_module, "GlobalSearch"),
        "LocalSearch": getattr(local_search_module, "LocalSearch"),
        "ContextBuilderResult": getattr(context_module, "ContextBuilderResult"),
        "GlobalContextBuilder": getattr(context_module, "GlobalContextBuilder"),
        "LocalContextBuilder": getattr(context_module, "LocalContextBuilder"),
        "NO_DATA_ANSWER": getattr(reduce_prompt_module, "NO_DATA_ANSWER"),
    }


class _SimpleTokenizer:
    def encode(self, text: str):
        if not text:
            return []
        return list(range(len(str(text).split())))


class _SimpleChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class _SimpleChoiceDelta:
    def __init__(self, content: str):
        self.content = content


class _SimpleResponseChoice:
    def __init__(self, content: str):
        self.message = _SimpleChoiceMessage(content)


class _SimpleChunkChoice:
    def __init__(self, content: str):
        self.delta = _SimpleChoiceDelta(content)


class _SimpleResponse:
    def __init__(self, content: str):
        self.choices = [_SimpleResponseChoice(content)]


class _OfficialCompletionBridge:
    """
    Minimal LLMCompletion-compatible object for official GraphRAG search classes.
    """

    def __init__(self, llm_client):
        self._llm = llm_client
        self.tokenizer = _SimpleTokenizer()

    async def completion_async(self, /, **kwargs):
        messages = kwargs.get("messages", [])
        stream = bool(kwargs.get("stream", False))
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)
        wants_json = bool(kwargs.get("response_format_json_object")) or (
            kwargs.get("response_format") is not None
        )

        if wants_json:
            payload = await self._llm.generate_json(messages, temperature=temperature)
            if not isinstance(payload, dict):
                payload = {"points": []}
            return _SimpleResponse(json.dumps(payload, ensure_ascii=False))

        text = await self._llm.generate_response(messages, temperature=temperature)
        if not isinstance(text, str):
            text = str(text or "")

        if stream:
            async def _gen():
                yield types.SimpleNamespace(choices=[_SimpleChunkChoice(text)])

            return _gen()

        return _SimpleResponse(text)

def _create_snapshot_builders(components: Dict[str, Any]):
    result_cls = components["ContextBuilderResult"]
    global_base = components["GlobalContextBuilder"]
    local_base = components["LocalContextBuilder"]

    class SnapshotGlobalContextBuilder(global_base):
        def __init__(self):
            self._chunks: List[str] = []
            self._records: Dict[str, Any] = {}

        def set_snapshot(self, chunks: List[str], records: Dict[str, Any]):
            self._chunks = chunks
            self._records = records

        async def build_context(self, query: str, conversation_history=None, **kwargs):
            _ = query, conversation_history, kwargs
            return result_cls(
                context_chunks=list(self._chunks),
                context_records=dict(self._records),
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
            )

    class SnapshotLocalContextBuilder(local_base):
        def __init__(self):
            self._chunk: str = ""
            self._records: Dict[str, Any] = {}

        def set_snapshot(self, chunk: str, records: Dict[str, Any]):
            self._chunk = chunk
            self._records = records

        def build_context(self, query: str, conversation_history=None, **kwargs):
            _ = query, conversation_history, kwargs
            return result_cls(
                context_chunks=self._chunk,
                context_records=dict(self._records),
                llm_calls=0,
                prompt_tokens=0,
                output_tokens=0,
            )

    return SnapshotGlobalContextBuilder, SnapshotLocalContextBuilder


class MSGraphRAGAdapter:
    """
    MS GraphRAG benchmark adapter:
    - Uses official GlobalSearch/LocalSearch search orchestration
    - Keeps current model endpoints (generation + rerank)
    - Keeps benchmark-compatible return type (answer, sources, trace)
    """

    def __init__(self, model_id: str = "local", corpus_tag: str = "default"):
        self.llm = get_llm_client(model_id)
        self.vllm = VLLMClient(model_name=model_id)
        self.neo4j = Neo4jService()
        self.corpus_tag = corpus_tag

        self.prefix = "MS_"
        import re as _re
        _safe_corpus = _re.sub(r"[^A-Za-z0-9_]", "_", self.corpus_tag)
        self.chunk_label = f"{self.prefix}{_safe_corpus}_Chunk"
        self.community_label = f"{self.prefix}{_safe_corpus}_Community"
        self.doc_label = f"{self.prefix}{_safe_corpus}_Document"
        self.vector_index = f"ms_graphrag_{_safe_corpus}_vector_idx"

        components = _load_official_ms_components()
        self._no_data_answer = components["NO_DATA_ANSWER"]

        global_builder_cls, local_builder_cls = _create_snapshot_builders(components)
        self._global_builder = global_builder_cls()
        self._local_builder = local_builder_cls()
        self._official_model = _OfficialCompletionBridge(self.llm)

        self._global_search = components["GlobalSearch"](
            model=self._official_model,
            context_builder=self._global_builder,
            response_type="single concise answer",
            map_max_length=260,
            reduce_max_length=320,
            map_llm_params={"temperature": 0.0},
            reduce_llm_params={"temperature": 0.1},
            concurrent_coroutines=8,
        )
        self._local_search = components["LocalSearch"](
            model=self._official_model,
            context_builder=self._local_builder,
            response_type="single concise answer",
            model_params={"temperature": 0.1},
        )
        self._agentic_runner = None
        self._agentic_full_service = None

    async def _get_community_summaries(self) -> List[Dict[str, Any]]:
        async with self.neo4j.driver.session() as session:
            query = f"""
                MATCH (c:{self.community_label})
                RETURN c.id AS id, c.summary AS summary, c.entities AS entities, c.title AS title
            """
            result = await session.run(query)  # type: ignore
            summaries = [dict(rec) async for rec in result]
            if summaries:
                return summaries

            fallback_query = f"""
                MATCH (d:{self.doc_label})
                WHERE d.summary IS NOT NULL
                RETURN d.filename AS id, d.title AS title, d.summary AS summary
                LIMIT 30
            """
            result = await session.run(fallback_query)  # type: ignore
            return [dict(rec) async for rec in result]

    @staticmethod
    def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body: List[str] = []
        for row in rows:
            values = []
            for col in columns:
                text = str(row.get(col, "")).replace("\n", " ").replace("|", " ")
                values.append(text[:1200])
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, sep] + body)

    async def _prepare_global_snapshot(self, query: str) -> List[Dict[str, Any]]:
        summaries = await self._get_community_summaries()
        if not summaries:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        query_embed_list = await self.vllm.get_embedding(query)
        if not query_embed_list:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        summary_texts = [str(s.get("summary", "") or "") for s in summaries if s.get("summary")]
        if not summary_texts:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        summary_embeds_list = await self.vllm.get_embeddings(summary_texts)
        if not summary_embeds_list:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        query_embed = np.array(query_embed_list)
        query_norm = float(np.linalg.norm(query_embed)) + 1e-8

        scored: List[Tuple[Dict[str, Any], float]] = []
        embed_idx = 0
        for item in summaries:
            summary = str(item.get("summary", "") or "")
            if not summary:
                continue
            summary_embed = np.array(summary_embeds_list[embed_idx])
            embed_idx += 1
            summary_norm = float(np.linalg.norm(summary_embed)) + 1e-8
            score = float(np.dot(query_embed, summary_embed) / (query_norm * summary_norm))
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        relevant = [item for item, _ in scored[:10]]
        if not relevant:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        context_chunks: List[str] = []
        batch_size = 5
        for offset in range(0, len(relevant), batch_size):
            batch = relevant[offset : offset + batch_size]
            rows = []
            for idx, item in enumerate(batch, start=1):
                rows.append({
                    "report_id": f"{offset + idx}",
                    "title": item.get("title", item.get("id", "Unknown")),
                    "summary": item.get("summary", ""),
                })
            table = self._to_markdown_table(rows, ["report_id", "title", "summary"])
            context_chunks.append(table)

        self._global_builder.set_snapshot(context_chunks, {"reports": relevant})
        return relevant

    async def _prepare_local_snapshot(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_embed = await self.vllm.get_embedding(query)
        if not query_embed:
            self._local_builder.set_snapshot("", {"sources": []})
            return []

        async with self.neo4j.driver.session() as session:
            query_cypher = f"""
                CALL db.index.vector.queryNodes('{self.vector_index}', $k, $embedding)
                YIELD node, score
                RETURN node.title AS title, node.sent_id AS sent_id, node.page AS page,
                       node.text AS text, node.source AS source, score
            """
            result = await session.run(query_cypher, {  # type: ignore
                "k": max(top_k * 3, 15),
                "embedding": query_embed,
            })
            nodes = [dict(rec) async for rec in result]

        if not nodes:
            self._local_builder.set_snapshot("", {"sources": []})
            return []

        texts = [str(n.get("text", "")) for n in nodes]
        scores = await self.vllm.rerank(query, texts)
        for idx, score in enumerate(scores):
            if idx < len(nodes):
                nodes[idx]["rerank_score"] = score
        nodes = sorted(nodes, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:top_k]

        rows = []
        for idx, node in enumerate(nodes, start=1):
            rows.append({
                "id": idx,
                "title": node.get("title", ""),
                "page": node.get("page", 0),
                "chunk": node.get("sent_id", 0),
                "text": str(node.get("text", ""))[:1400],
            })
        context_table = self._to_markdown_table(rows, ["id", "title", "page", "chunk", "text"])
        self._local_builder.set_snapshot(context_table, {"sources": nodes})
        return nodes

    async def global_search(self, query: str) -> Tuple[str, List, List]:
        relevant = await self._prepare_global_snapshot(query)
        if not relevant:
            return self._no_data_answer, [], []

        result = await self._global_search.search(query)
        answer = str(result.response or "").strip() or self._no_data_answer
        sources = [
            {
                "doc": item.get("title", item.get("id", "Community")),
                "page": 0,
                "text": item.get("summary", ""),
                "sent_id": 0,
            }
            for item in relevant
        ]
        trace = [{
            "step": "ms_global_official_search",
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
        }]
        return answer, sources, trace

    async def local_search(self, query: str, top_k: int = 5) -> Tuple[str, List, List]:
        nodes = await self._prepare_local_snapshot(query, top_k=top_k)
        if not nodes:
            return "", [], []

        result = await self._local_search.search(query)
        answer = str(result.response or "").strip()
        sources = [
            {
                "doc": n.get("title", ""),
                "page": n.get("page", 0),
                "text": n.get("text", ""),
                "sent_id": n.get("sent_id", 0),
            }
            for n in nodes
        ]
        trace = [{
            "step": "ms_local_official_search",
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
        }]
        return answer, sources, trace

    async def retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        nodes = await self._prepare_local_snapshot(query, top_k=top_k)
        if not nodes:
            return "", []

        blocks: List[str] = []
        for node in nodes:
            title = str(node.get("title", "") or "")
            page = int(node.get("page") or 0)
            sent_id = int(node.get("sent_id") or 0)
            text = str(node.get("text", "") or "")
            blocks.append(f"[[{title}, Page {page}, Chunk {sent_id}]]\n{text}")
        return "\n\n---\n\n".join(blocks), nodes

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
                        strategy_name="ms_graphrag",
                        top_k=5,
                    )
                return await self._agentic_runner.run(query, history)

            if self._agentic_full_service is None:
                from models.agentic_core import RetrievalGraphAdapter
                from models.hyporeflect.service import AgentService

                graph_adapter = RetrievalGraphAdapter(
                    backend=self,
                    strategy_name="ms_graphrag",
                    default_top_k=5,
                )
                self._agentic_full_service = AgentService(
                    strategy="hyporeflect",
                    corpus_tag=self.corpus_tag,
                    llm_override=self.llm,
                    grag_override=graph_adapter,
                )
            return await self._agentic_full_service.run_workflow(query, history)

        _ = history
        abstract_keywords = [
            "overall",
            "summary",
            "main themes",
            "in general",
            "relationship between",
            "high-level",
            "broadly",
            "across documents",
        ]
        is_global = any(kw in query.lower() for kw in abstract_keywords)
        if is_global:
            logger.info("MS GraphRAG official GlobalSearch path")
            return await self.global_search(query)
        logger.info("MS GraphRAG official LocalSearch path")
        return await self.local_search(query)
