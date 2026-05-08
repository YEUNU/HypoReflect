"""Cross-encoder reranking (paper §3.2.3).

Reranker score combined with meta boost and boilerplate penalty:
final_score = rerank + W_meta * meta_boost - W_boilerplate * boilerplate_penalty

Threshold tau_r = RAGConfig.RERANKER_THRESHOLD (default 0.5 in paper).
"""
import logging
from typing import Any

from core.config import RAGConfig
from utils.prompts import (
    RERANKER_INSTRUCTION,
    RERANK_QUERY_SIMPLIFY_FORMAT_INSTRUCTION,
    RERANK_QUERY_SIMPLIFY_PROMPT,
)


logger = logging.getLogger(__name__)


class RerankMixin:
    @staticmethod
    def _reranker_instruction() -> str:
        return RERANKER_INSTRUCTION

    async def _simplified_rerank_query(self, query: str) -> str:
        """Strip verbose preludes/role-framing/output-format instructions
        from a user query before handing it to the cross-encoder reranker.
        Long, role-played queries silently collapse reranker scores (verified
        empirically: same chunk drops from 0.94 to 0.03 when the query is
        wrapped in "Answer as if you are an equity research analyst...").
        Cached per-query on the GraphRAG instance to avoid repeating the
        LLM call across multi-turn retrievals.
        """
        original = str(query or "").strip()
        if not original:
            return original
        # Skip the LLM call for short queries (no meaningful prelude to strip).
        if len(original) <= 80:
            return original
        cache = getattr(self, "_simplified_rerank_query_cache", None)
        if cache is None:
            cache = {}
            self._simplified_rerank_query_cache = cache
        if original in cache:
            return cache[original]
        try:
            response = await self.llm.generate_json(
                [
                    {"role": "user", "content": RERANK_QUERY_SIMPLIFY_PROMPT.format(query=original)},
                    {"role": "user", "content": RERANK_QUERY_SIMPLIFY_FORMAT_INSTRUCTION},
                ]
            )
            simplified = str((response or {}).get("question", "") or "").strip()
        except Exception as exc:
            logger.warning("Rerank query simplification failed: %s", exc)
            simplified = ""
        if not simplified:
            simplified = original
        cache[original] = simplified
        return simplified

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
        rerank_query = await self._simplified_rerank_query(query)
        try:
            scores = await self.llm.rerank(rerank_query, doc_texts, instruction=self._reranker_instruction())
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
            # Strict filter: when the query is anchored to a company, drop
            # cross-company chunks entirely instead of merely demoting them.
            # The previous "demote, don't drop" behavior let cross-company
            # chunks survive past top_k and get cited by the synthesis stage
            # (e.g., AMD content cited under an AMEX query). Fall back to the
            # demote-only behavior if strict filtering would empty the pool.
            matched = [node for node in reranked_nodes if self._node_matches_company(node, query_meta)]
            if matched:
                reranked_nodes = matched
            else:
                logger.info(
                    "Company filter would empty candidate pool (keys=%s); "
                    "falling back to demote-only ordering.",
                    sorted(company_keys),
                )

        final_nodes = [
            node for node in reranked_nodes
            if node.get("rerank_score", 0.0) >= RAGConfig.RERANKER_THRESHOLD
        ][:top_k]
        # Top up to top_k with the next-best ungated candidates when the
        # reranker gate filters too aggressively. The previous behavior
        # ("if EMPTY, fall back; else keep gated as-is") underfilled the
        # candidate pool whenever a single chunk crossed tau_r — in
        # practice this collapsed bootstrap retrieval to ~1 result on many
        # queries, even with top_k=12.
        if len(final_nodes) < top_k and len(final_nodes) < len(reranked_nodes):
            seen = {id(node) for node in final_nodes}
            for node in reranked_nodes:
                if id(node) in seen:
                    continue
                final_nodes.append(node)
                seen.add(id(node))
                if len(final_nodes) >= top_k:
                    break
        return final_nodes, reranked_nodes

    async def hybrid_search(self, query: str, top_k: int = 5) -> tuple:
        nodes = await self._hybrid_rrf_candidates(query, limit=max(20, top_k * 4), channel="body")
        if not nodes:
            return "", []

        doc_texts = [node["text"] for node in nodes]
        rerank_query = await self._simplified_rerank_query(query)
        try:
            scores = await self.llm.rerank(rerank_query, doc_texts, instruction=self._reranker_instruction())
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
        # Top up to top_k with next-best ungated candidates (same rationale
        # as `_rerank_and_select`).
        if len(gated_nodes) < top_k and len(gated_nodes) < len(reranked_nodes):
            seen = {id(node) for node in gated_nodes}
            for node in reranked_nodes:
                if id(node) in seen:
                    continue
                gated_nodes.append(node)
                seen.add(id(node))
                if len(gated_nodes) >= top_k:
                    break
        if not gated_nodes:
            return "", []
        context = self._build_context_from_nodes(gated_nodes)
        return context, gated_nodes
