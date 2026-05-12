"""
[MS GraphRAG] adapter wired to official GlobalSearch/LocalSearch classes.

Reads parquet artifacts produced by official_indexer.py (data/ms_graphrag_output/
<corpus_tag>/{entities,communities,community_reports,text_units}.parquet) and
hands snapshots to upstream Microsoft query orchestration. Models route through
our local vLLM via the existing VLLMClient.
"""

import importlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.vllm_client import VLLMClient, get_llm_client
from models.ms_graphrag.official_indexer import output_dir_for

logger = logging.getLogger(__name__)


def _load_official_ms_components() -> Dict[str, Any]:
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
        return [ord(ch) for ch in str(text or "")]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if isinstance(i, int))


class _SimpleResponseChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})()


class _SimpleStreamChunkChoice:
    def __init__(self, content: str):
        self.delta = type("D", (), {"content": content})()


class _SimpleStreamChunk:
    def __init__(self, content: str):
        self.choices = [_SimpleStreamChunkChoice(content)]


class _SimpleResponse:
    """Behaves as both a non-streaming response (.choices[0].message.content)
    AND an async iterator (yields one chunk with full content). graphrag 3.0.1
    LocalSearch.search uses `async for chunk in response`; GlobalSearch uses
    response.choices[0].message.content directly. One class covers both."""

    def __init__(self, content: str):
        self.choices = [_SimpleResponseChoice(content)]
        self.content = content

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        yield _SimpleStreamChunk(self.content)


class _OfficialCompletionBridge:
    """Minimal LLMCompletion-compatible adapter for the official GraphRAG search classes.

    Library quirk: GlobalSearch._map_response_single_batch passes
    response_format_json_object=True both explicitly AND via **map_llm_params
    (which it pre-populated when json_mode=True). That raises a duplicate-kwarg
    TypeError. We swallow the duplicate at the bridge.
    """

    def __init__(self, llm_client):
        self._llm = llm_client
        self.tokenizer = _SimpleTokenizer()

    async def completion_async(self, /, **kwargs):
        kwargs.pop("response_format", None)
        wants_json = bool(kwargs.pop("response_format_json_object", False))

        messages = kwargs.get("messages", [])
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)

        if wants_json:
            payload = await self._llm.generate_json(messages, temperature=temperature)
            if not isinstance(payload, dict):
                payload = {"points": []}
            import json as _json
            return _SimpleResponse(_json.dumps(payload, ensure_ascii=False))

        text = await self._llm.generate_response(messages, temperature=temperature)
        return _SimpleResponse(str(text or ""))


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
                llm_calls=0, prompt_tokens=0, output_tokens=0,
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
                llm_calls=0, prompt_tokens=0, output_tokens=0,
            )

    return SnapshotGlobalContextBuilder, SnapshotLocalContextBuilder


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            text = str(row.get(col, "")).replace("\n", " ").replace("|", " ")
            values.append(text[:1200])
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + body)


class MSGraphRAGAdapter:
    """MS GraphRAG benchmark adapter (parquet-backed).

    Indexing artifacts come from official MS pipeline (run_official_index in
    official_indexer.py). Query path uses official GlobalSearch/LocalSearch
    with snapshot builders fed by parquet + on-the-fly embedding.
    """

    def __init__(self, model_id: str = "local", corpus_tag: str = "default"):
        self.llm = get_llm_client(model_id)
        self.vllm = VLLMClient(model_name=model_id)
        self.corpus_tag = corpus_tag
        self.output_dir = output_dir_for(corpus_tag)

        # Lazy-load parquet on first query.
        self._community_reports: Optional[pd.DataFrame] = None
        self._entities: Optional[pd.DataFrame] = None
        self._text_units: Optional[pd.DataFrame] = None
        self._text_unit_embeds: Optional[np.ndarray] = None  # cached corpus embeddings

        components = _load_official_ms_components()
        self._no_data_answer = components["NO_DATA_ANSWER"]

        global_builder_cls, local_builder_cls = _create_snapshot_builders(components)
        self._global_builder = global_builder_cls()
        self._local_builder = local_builder_cls()
        self._official_model = _OfficialCompletionBridge(self.llm)

        # json_mode=False: bridge handles JSON unconditionally and avoids the
        # library's response_format_json_object duplicate-kwarg trap.
        self._global_search = components["GlobalSearch"](
            model=self._official_model,
            context_builder=self._global_builder,
            response_type="single concise answer",
            map_max_length=260,
            reduce_max_length=320,
            map_llm_params={"temperature": 0.0},
            reduce_llm_params={"temperature": 0.1},
            concurrent_coroutines=8,
            json_mode=False,
        )
        self._local_search = components["LocalSearch"](
            model=self._official_model,
            context_builder=self._local_builder,
            response_type="single concise answer",
            model_params={"temperature": 0.1},
        )
        self._agentic_runner = None
        self._agentic_full_service = None

    # ------------------------------------------------------------------ parquet I/O

    def _read_parquet(self, name: str) -> pd.DataFrame:
        path = self.output_dir / f"{name}.parquet"
        if not path.exists():
            logger.warning("MS parquet missing: %s", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _ensure_loaded(self) -> None:
        if self._community_reports is None:
            self._community_reports = self._read_parquet("community_reports")
        if self._entities is None:
            self._entities = self._read_parquet("entities")
        if self._text_units is None:
            self._text_units = self._read_parquet("text_units")

    # ------------------------------------------------------------------ snapshots

    async def _get_community_summaries(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        df = self._community_reports
        if df is None or df.empty:
            return []
        # Newest schema: id, community, level, title, summary, full_content, rank, ...
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row.get("community", row.get("id", ""))),
                "title": str(row.get("title", "") or ""),
                "summary": str(row.get("summary", "") or row.get("full_content", "") or ""),
                "rank": float(row.get("rank", 0.0) or 0.0),
                "level": int(row.get("level", 0) or 0),
            })
        return records

    async def _prepare_global_snapshot(self, query: str) -> List[Dict[str, Any]]:
        summaries = await self._get_community_summaries()
        if not summaries:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        query_embed_list = await self.vllm.get_embedding(query)
        if not query_embed_list:
            # Fall back to rank-only ordering if embedding unavailable.
            relevant = sorted(summaries, key=lambda s: s.get("rank", 0.0), reverse=True)[:10]
        else:
            summary_texts = [s["summary"] for s in summaries if s["summary"]]
            summary_embeds = await self.vllm.get_embeddings(summary_texts)
            if not summary_embeds:
                relevant = sorted(summaries, key=lambda s: s.get("rank", 0.0), reverse=True)[:10]
            else:
                qv = np.array(query_embed_list, dtype=np.float32)
                qn = float(np.linalg.norm(qv)) + 1e-8
                scored: List[Tuple[Dict[str, Any], float]] = []
                ei = 0
                for item in summaries:
                    if not item["summary"]:
                        continue
                    sv = np.array(summary_embeds[ei], dtype=np.float32)
                    ei += 1
                    sn = float(np.linalg.norm(sv)) + 1e-8
                    sim = float(np.dot(qv, sv) / (qn * sn))
                    # Combine cosine sim with community rank (light weighting).
                    scored.append((item, sim + 0.05 * item.get("rank", 0.0)))
                scored.sort(key=lambda x: x[1], reverse=True)
                relevant = [item for item, _ in scored[:10]]

        if not relevant:
            self._global_builder.set_snapshot([], {"reports": []})
            return []

        # Pack in batches of 5 for the global Map step.
        context_chunks: List[str] = []
        batch_size = 5
        for offset in range(0, len(relevant), batch_size):
            batch = relevant[offset : offset + batch_size]
            rows = []
            for idx, item in enumerate(batch, start=1):
                rows.append({
                    "report_id": str(offset + idx),
                    "title": item["title"] or item["id"],
                    "summary": item["summary"],
                })
            context_chunks.append(_to_markdown_table(rows, ["report_id", "title", "summary"]))

        self._global_builder.set_snapshot(context_chunks, {"reports": relevant})
        return relevant

    async def _prepare_local_snapshot(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Local search: embed all text_units (one-time), cosine-rank against
        the query, then rerank top candidates and feed top_k to LocalSearch.
        """
        self._ensure_loaded()
        df = self._text_units
        if df is None or df.empty:
            self._local_builder.set_snapshot("", {"sources": []})
            return []

        query_embed = await self.vllm.get_embedding(query)
        if not query_embed:
            self._local_builder.set_snapshot("", {"sources": []})
            return []

        if self._text_unit_embeds is None:
            texts = [str(t or "") for t in df["text"].tolist()]
            embeds = await self.vllm.get_embeddings(texts)
            self._text_unit_embeds = np.array(embeds, dtype=np.float32)
            logger.info(
                "MS local-snapshot: cached %d text_unit embeddings for %s",
                len(texts), self.corpus_tag,
            )

        qv = np.array(query_embed, dtype=np.float32)
        qn = float(np.linalg.norm(qv)) + 1e-8
        sims = self._text_unit_embeds @ qv / (
            np.linalg.norm(self._text_unit_embeds, axis=1) * qn + 1e-8
        )
        ann_k = max(top_k * 3, 15)
        top_idx = np.argsort(-sims)[:ann_k]
        cand = df.iloc[top_idx].copy()
        cand["ann_score"] = sims[top_idx]

        texts = cand["text"].astype(str).tolist()
        rerank_scores = await self.vllm.rerank(query, texts)
        cand["rerank_score"] = rerank_scores
        cand = cand.sort_values("rerank_score", ascending=False).head(top_k)

        nodes: List[Dict[str, Any]] = []
        for _, r in cand.iterrows():
            nodes.append({
                "title": str(r.get("document_id", "") or "")[:80],
                "page": 0,
                "sent_id": int(r.get("human_readable_id", 0) or 0),
                "text": str(r.get("text", "") or ""),
                "source": str(r.get("document_id", "") or ""),
                "rerank_score": float(r.get("rerank_score", 0.0) or 0.0),
            })

        rows = []
        for idx, n in enumerate(nodes, start=1):
            rows.append({
                "id": idx,
                "title": n["title"],
                "page": n["page"],
                "chunk": n["sent_id"],
                "text": n["text"][:1400],
            })
        context_table = _to_markdown_table(rows, ["id", "title", "page", "chunk", "text"])
        self._local_builder.set_snapshot(context_table, {"sources": nodes})
        return nodes

    # ------------------------------------------------------------------ search APIs

    async def global_search(self, query: str) -> Tuple[str, List, List]:
        relevant = await self._prepare_global_snapshot(query)
        if not relevant:
            return self._no_data_answer, [], []

        result = await self._global_search.search(query)
        answer = str(result.response or "").strip() or self._no_data_answer
        sources = [{
            "doc": item["title"] or item["id"],
            "page": 0,
            "text": item["summary"],
            "sent_id": 0,
        } for item in relevant]
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
        sources = [{
            "doc": n["title"], "page": n["page"],
            "text": n["text"], "sent_id": n["sent_id"],
        } for n in nodes]
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
        blocks = []
        for n in nodes:
            blocks.append(f"[[{n['title']}, Page {n['page']}, Chunk {n['sent_id']}]]\n{n['text']}")
        return "\n\n---\n\n".join(blocks), nodes

    async def run_workflow(self, query: str, history: Optional[List[Dict]] = None) -> Tuple[str, List, List]:
        agentic_mode = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
        if agentic_mode == "on":
            agentic_pipeline = str(os.environ.get("RAG_AGENTIC_PIPELINE", "full") or "full").strip().lower()
            if agentic_pipeline == "lite":
                if self._agentic_runner is None:
                    from models.agentic_core import AgenticOrchestrator
                    self._agentic_runner = AgenticOrchestrator(
                        llm=self.llm, backend=self,
                        strategy_name="ms_graphrag", top_k=5,
                    )
                return await self._agentic_runner.run(query, history)

            if self._agentic_full_service is None:
                from models.agentic_core import RetrievalGraphAdapter
                from models.hyporeflect.service import AgentService
                graph_adapter = RetrievalGraphAdapter(
                    backend=self, strategy_name="ms_graphrag", default_top_k=5,
                )
                self._agentic_full_service = AgentService(
                    strategy="hyporeflect", corpus_tag=self.corpus_tag,
                    llm_override=self.llm, grag_override=graph_adapter,
                )
            return await self._agentic_full_service.run_workflow(query, history)

        _ = history
        abstract_keywords = [
            "overall", "summary", "main themes", "in general",
            "relationship between", "high-level", "broadly", "across documents",
        ]
        is_global = any(kw in query.lower() for kw in abstract_keywords)
        if is_global:
            logger.info("MS GraphRAG official GlobalSearch path")
            return await self.global_search(query)
        logger.info("MS GraphRAG official LocalSearch path")
        return await self.local_search(query)
