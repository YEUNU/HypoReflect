"""Cross-encoder reranking (paper §3.2.3).

Reranker score combined with meta boost and boilerplate penalty:
final_score = rerank + W_meta * meta_boost - W_boilerplate * boilerplate_penalty

Threshold tau_r = RAGConfig.RERANKER_THRESHOLD (default 0.5 in paper).
"""
import logging
from typing import Any

from core.config import RAGConfig
from utils.prompts import RERANKER_INSTRUCTION


logger = logging.getLogger(__name__)


class RerankMixin:
    @staticmethod
    def _reranker_instruction() -> str:
        return RERANKER_INSTRUCTION

    async def _rerank_and_select(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        query_meta: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            return [], []

        self._apply_retrieval_calibration(candidates, query_meta)
        doc_texts = [node.get("text", "") for node in candidates]
        try:
            scores = await self.llm.rerank(query, doc_texts, instruction=self._reranker_instruction())
        except Exception as error:
            logger.warning("Retrieve reranking failed: %s", error)
            scores = [0.0] * len(candidates)

        for index, score in enumerate(scores):
            candidates[index]["rerank_score"] = score
            candidates[index]["final_score"] = (
                score
                + (RAGConfig.META_BOOST_WEIGHT * candidates[index].get("meta_boost", 0.0))
                - (RAGConfig.BOILERPLATE_PENALTY_WEIGHT * candidates[index].get("boilerplate_penalty", 0.0))
            )

        reranked_nodes = sorted(candidates, key=lambda item: item.get("final_score", 0.0), reverse=True)
        company_keys = set(query_meta.get("company_keys") or [])
        if company_keys:
            matched_first = [node for node in reranked_nodes if self._node_matches_company(node, query_meta)]
            if matched_first:
                reranked_nodes = matched_first + [node for node in reranked_nodes if not self._node_matches_company(node, query_meta)]

        final_nodes = [
            node for node in reranked_nodes
            if node.get("rerank_score", 0.0) >= RAGConfig.RERANKER_THRESHOLD
        ][:top_k]
        if not final_nodes and reranked_nodes:
            final_nodes = reranked_nodes[:top_k]
        return final_nodes, reranked_nodes

    async def hybrid_search(self, query: str, top_k: int = 5) -> tuple:
        nodes = await self._hybrid_rrf_candidates(query, limit=max(20, top_k * 4), channel="body")
        if not nodes:
            return "", []

        doc_texts = [node["text"] for node in nodes]
        try:
            scores = await self.llm.rerank(query, doc_texts, instruction=self._reranker_instruction())
        except Exception as error:
            logger.warning("Hybrid search reranking failed: %s", error)
            scores = [0.0] * len(nodes)

        for index, score in enumerate(scores):
            nodes[index]["rerank_score"] = score
            nodes[index]["final_score"] = (
                score
                + (RAGConfig.META_BOOST_WEIGHT * nodes[index].get("meta_boost", 0.0))
                - (RAGConfig.BOILERPLATE_PENALTY_WEIGHT * nodes[index].get("boilerplate_penalty", 0.0))
            )

        reranked_nodes = sorted(nodes, key=lambda item: item.get("final_score", 0.0), reverse=True)
        gated_nodes = [node for node in reranked_nodes if node.get("rerank_score", 0.0) >= RAGConfig.RERANKER_THRESHOLD][:top_k]
        if not gated_nodes and reranked_nodes:
            logger.info("Hybrid reranker gate removed all candidates; using top reranked nodes without threshold.")
            gated_nodes = reranked_nodes[:top_k]
        if not gated_nodes:
            return "", []
        context = self._build_context_from_nodes(gated_nodes)
        return context, gated_nodes
