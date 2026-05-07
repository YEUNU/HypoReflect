"""Two-stage Q-/Q+ retrieval entry point (paper §3.2.3).

Stage 1 — grounded retrieval: RRF over body (weight 0.7) and Q- (weight 0.3).
   This deviates from the paper's nominal Q-=0.7/body=0.3 weighting; raw
   chunk text is empirically more reliable than LLM-generated Q- queries
   under the current index, where Q- regeneration could not match per-paper
   indexing quality.
Stage 2 — Q+ expansion: triggered when Stage 1 returns fewer candidates than
   the slot budget, or when the best rerank score is too close to tau_r.
   Adds Q+ (weight 0.6) plus a Q- support pool (weight 0.4) and re-ranks
   once more.

Query rewriting (N_r=2, w_r=0.85) supplies additional query variants whose
RRF contributions are scaled by w_r relative to the original query.
"""
from typing import Any

from core.config import RAGConfig


class RetrieveMixin:
    async def retrieve(self, query: str, top_k: int = 5) -> tuple:
        rewrites: list[str] = []
        if RAGConfig.ENABLE_QUERY_REWRITE:
            rewrites = await self._rewrite_query(query)
        query_variants = [query] + rewrites[: max(0, RAGConfig.QUERY_REWRITE_COUNT)]

        stage1_merged: dict[str, dict[str, Any]] = {}
        candidate_limit_per_query = max(20, top_k * 8)

        def _accumulate(
            merged: dict[str, dict[str, Any]],
            nodes: list[dict[str, Any]],
            score_key: str,
            weight: float,
        ) -> None:
            for rank, node in enumerate(nodes):
                node_id = self._node_identity(node)
                if node_id not in merged:
                    item = dict(node)
                    item.setdefault("stage1_rrf_score", 0.0)
                    item.setdefault("stage2_rrf_score", 0.0)
                    item.setdefault("stage2_support_score", 0.0)
                    merged[node_id] = item
                merged[node_id][score_key] += weight * (1.0 / (RAGConfig.RRF_K_CONSTANT + rank))

        # --- Stage 1: grounded retrieval (Q- 0.7 + body 0.3) ---
        for index, query_text in enumerate(query_variants):
            query_weight = 1.0 if index == 0 else RAGConfig.QUERY_REWRITE_WEIGHT
            if RAGConfig.ABLATION_Q_MINUS:
                q_minus_nodes = await self._hybrid_rrf_candidates(query_text, limit=candidate_limit_per_query, channel="q_minus")
                body_nodes = await self._hybrid_rrf_candidates(query_text, limit=max(10, top_k * 4), channel="body")
                _accumulate(stage1_merged, q_minus_nodes, "stage1_rrf_score", query_weight * 0.3)
                _accumulate(stage1_merged, body_nodes, "stage1_rrf_score", query_weight * 0.7)
            else:
                body_nodes = await self._hybrid_rrf_candidates(query_text, limit=candidate_limit_per_query, channel="body")
                _accumulate(stage1_merged, body_nodes, "stage1_rrf_score", query_weight * 1.0)

        if not stage1_merged:
            return "", []

        stage1_candidates = sorted(
            stage1_merged.values(),
            key=lambda item: item.get("stage1_rrf_score", 0.0),
            reverse=True,
        )[: max(20, top_k * 6)]

        query_meta = self._extract_query_metadata(query)
        stage1_nodes, stage1_reranked = await self._rerank_and_select(query, stage1_candidates, top_k, query_meta)

        if not stage1_nodes and not stage1_reranked:
            return "", []

        best_stage1_score = stage1_nodes[0].get("rerank_score", 0.0) if stage1_nodes else 0.0
        need_expand = (len(stage1_nodes) < top_k) or (best_stage1_score < (RAGConfig.RERANKER_THRESHOLD + 0.08))

        if not RAGConfig.ABLATION_Q_PLUS:
            need_expand = False

        if not need_expand:
            for node in stage1_nodes:
                node.pop("stage1_rrf_score", None)
                node.pop("stage2_rrf_score", None)
                node.pop("stage2_support_score", None)
            return self._build_context_from_nodes(stage1_nodes), stage1_nodes

        # --- Stage 2: Q+ expansion (Q+ 0.6 + Q- support 0.4) ---
        expanded: dict[str, dict[str, Any]] = {self._node_identity(node): dict(node) for node in stage1_candidates}
        q_plus_weight = 0.3
        q_minus_support_weight = 0.7
        for index, query_text in enumerate(query_variants):
            query_weight = 1.0 if index == 0 else RAGConfig.QUERY_REWRITE_WEIGHT
            q_plus_nodes = await self._hybrid_rrf_candidates(query_text, limit=candidate_limit_per_query, channel="q_plus")
            q_minus_support_nodes = await self._hybrid_rrf_candidates(query_text, limit=max(10, top_k * 4), channel="q_minus")
            _accumulate(expanded, q_plus_nodes, "stage2_rrf_score", query_weight * q_plus_weight)
            _accumulate(expanded, q_minus_support_nodes, "stage2_support_score", query_weight * q_minus_support_weight)

        if not expanded:
            return "", []

        for node in expanded.values():
            node["hybrid_rrf_score"] = (
                node.get("stage1_rrf_score", 0.0)
                + node.get("stage2_rrf_score", 0.0)
                + node.get("stage2_support_score", 0.0)
            )

        expanded_candidates = sorted(
            expanded.values(),
            key=lambda item: item.get("hybrid_rrf_score", 0.0),
            reverse=True,
        )[: max(24, top_k * 8)]

        final_nodes, _ = await self._rerank_and_select(query, expanded_candidates, top_k, query_meta)
        if not final_nodes:
            final_nodes = stage1_nodes or stage1_reranked[:top_k]
        if not final_nodes:
            return "", []

        for node in final_nodes:
            node.pop("stage1_rrf_score", None)
            node.pop("stage2_rrf_score", None)
            node.pop("stage2_support_score", None)
            node.pop("hybrid_rrf_score", None)
        return self._build_context_from_nodes(final_nodes), final_nodes
