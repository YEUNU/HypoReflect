"""
[MS GraphRAG] adapter using official graphrag.api Python interface.

Uses graphrag.api.local_search / global_search (graphrag==3.0.1) which performs
the full KG-grounded search: entity embedding retrieval → entity/relationship/
community context + text_units → LLM answer.

Parquet + lancedb artifacts are read from data/ms_graphrag_output/<corpus_tag>/
as built by official_indexer.py. No re-indexing needed.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.vllm_client import VLLMClient, get_llm_client
from models.ms_graphrag.official_indexer import (
    build_config,
    input_dir_for,
    output_dir_for,
)

logger = logging.getLogger(__name__)


class MSGraphRAGAdapter:
    def __init__(self, model_id: str = "local", corpus_tag: str = "default"):
        self.llm = get_llm_client(model_id)
        self.vllm = VLLMClient(model_name=model_id)
        self.corpus_tag = corpus_tag
        self.output_dir = output_dir_for(corpus_tag)

        # GraphRagConfig pointing to local vLLM + lancedb at output_dir
        self._config = build_config(corpus_tag, input_dir_for(corpus_tag))

        # Lazy-loaded parquet DataFrames
        self._entities: Optional[pd.DataFrame] = None
        self._communities: Optional[pd.DataFrame] = None
        self._community_reports: Optional[pd.DataFrame] = None
        self._text_units: Optional[pd.DataFrame] = None
        self._relationships: Optional[pd.DataFrame] = None
        self._documents: Optional[pd.DataFrame] = None
        self._doc_id_to_title: Optional[Dict[str, str]] = None
        self._short_id_to_doc_id: Optional[Dict[str, str]] = None

        # ANN embedding cache for retrieve() (agentic backend)
        self._text_unit_embeds = None

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
        if self._entities is None:
            self._entities = self._read_parquet("entities")
        if self._communities is None:
            self._communities = self._read_parquet("communities")
        if self._community_reports is None:
            self._community_reports = self._read_parquet("community_reports")
        if self._text_units is None:
            self._text_units = self._read_parquet("text_units")
            # Build short_id → document_id lookup for source extraction
            if not self._text_units.empty and "human_readable_id" in self._text_units.columns:
                self._short_id_to_doc_id = {
                    str(row["human_readable_id"]): str(row.get("document_id", "") or "")
                    for _, row in self._text_units.iterrows()
                }
            else:
                self._short_id_to_doc_id = {}
        if self._relationships is None:
            self._relationships = self._read_parquet("relationships")
        if self._documents is None:
            self._documents = self._read_parquet("documents")
            if not self._documents.empty and "id" in self._documents.columns:
                self._doc_id_to_title = {
                    str(row["id"]): str(row.get("title", "") or "")
                    for _, row in self._documents.iterrows()
                }
            else:
                self._doc_id_to_title = {}

    # ------------------------------------------------------------------ source extraction

    def _extract_sources(self, context_data: Any) -> List[Dict[str, Any]]:
        """Extract source nodes from local_search context_records for doc_match metrics.

        context_records["sources"] is a DataFrame with columns [id, text, ...] where
        id == text_unit.short_id (== str(human_readable_id)).
        """
        sources = []
        try:
            if not isinstance(context_data, dict):
                return sources
            src_df = context_data.get("sources")
            if src_df is None or not hasattr(src_df, "iterrows") or src_df.empty:
                return sources
            short_id_map = self._short_id_to_doc_id or {}
            doc_map = self._doc_id_to_title or {}
            for _, row in src_df.iterrows():
                unit_id = str(row.get("id", "") or "")
                doc_id = short_id_map.get(unit_id, "")
                title = doc_map.get(doc_id, doc_id)
                title = re.sub(r"\.(pdf|txt|md|json)$", "", title, flags=re.IGNORECASE)
                sources.append({
                    "doc": title,
                    "page": 0,
                    "text": str(row.get("text", "") or ""),
                    "sent_id": 0,
                })
        except Exception as exc:
            logger.debug("Could not extract sources from context_data: %s", exc)
        return sources

    # ------------------------------------------------------------------ search APIs

    async def local_search(self, query: str) -> Tuple[str, List, List]:
        import graphrag.api as gapi
        self._ensure_loaded()

        response, context_data = await gapi.local_search(
            config=self._config,
            entities=self._entities,
            communities=self._communities,
            community_reports=self._community_reports,
            text_units=self._text_units,
            relationships=self._relationships,
            covariates=None,
            community_level=2,
            response_type="single concise answer",
            query=query,
        )

        answer = str(response or "").strip()
        sources = self._extract_sources(context_data)
        trace = [{"step": "ms_local_search_api"}]
        return answer, sources, trace

    async def global_search(self, query: str) -> Tuple[str, List, List]:
        import graphrag.api as gapi
        self._ensure_loaded()

        response, context_data = await gapi.global_search(
            config=self._config,
            entities=self._entities,
            communities=self._communities,
            community_reports=self._community_reports,
            community_level=2,
            dynamic_community_selection=False,
            response_type="single concise answer",
            query=query,
        )

        answer = str(response or "").strip()
        sources = self._extract_sources(context_data)
        trace = [{"step": "ms_global_search_api"}]
        return answer, sources, trace

    async def retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Dense ANN retrieval for agentic mode backend."""
        import numpy as np
        self._ensure_loaded()
        df = self._text_units
        if df is None or df.empty:
            return "", []

        query_embed = await self.vllm.get_embedding(query)
        if not query_embed:
            return "", []

        if self._text_unit_embeds is None:
            texts = [str(t or "") for t in df["text"].tolist()]
            embeds = await self.vllm.get_embeddings(texts)
            self._text_unit_embeds = np.array(embeds, dtype=np.float32)
            logger.info("MS retrieve: cached %d text_unit embeddings", len(texts))

        qv = np.array(query_embed, dtype=np.float32)
        qn = float(np.linalg.norm(qv)) + 1e-8
        sims = self._text_unit_embeds @ qv / (
            np.linalg.norm(self._text_unit_embeds, axis=1) * qn + 1e-8
        )
        ann_k = max(top_k * 3, 15)
        top_idx = sims.argsort()[::-1][:ann_k]
        cand = df.iloc[top_idx].copy()
        cand["ann_score"] = sims[top_idx]

        texts = cand["text"].astype(str).tolist()
        rerank_scores = await self.vllm.rerank(query, texts)
        cand["rerank_score"] = rerank_scores
        cand = cand.sort_values("rerank_score", ascending=False).head(top_k)

        doc_map = self._doc_id_to_title or {}
        nodes = []
        for _, r in cand.iterrows():
            raw_id = str(r.get("document_id", "") or "")
            title = doc_map.get(raw_id, raw_id)
            title = re.sub(r"\.(pdf|txt|md|json)$", "", title, flags=re.IGNORECASE)
            nodes.append({
                "title": title,
                "page": 0,
                "sent_id": int(r.get("human_readable_id", 0) or 0),
                "text": str(r.get("text", "") or ""),
                "source": raw_id,
                "rerank_score": float(r.get("rerank_score", 0.0) or 0.0),
            })

        blocks = [
            f"[[{n['title']}, Page {n['page']}, Chunk {n['sent_id']}]]\n{n['text']}"
            for n in nodes
        ]
        return "\n\n---\n\n".join(blocks), nodes

    # ------------------------------------------------------------------ workflow

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
            logger.info("MS GraphRAG API GlobalSearch path")
            return await self.global_search(query)
        logger.info("MS GraphRAG API LocalSearch path")
        return await self.local_search(query)
