"""Two-stage hybrid retrieval with RRF (paper §3.2.3).

Each retrieval round combines vector and fulltext search via reciprocal rank
fusion: RRF(d) = w_v / (k + r_v(d)) + w_t / (k + r_t(d))
with k = RAGConfig.RRF_K_CONSTANT (60), w_v = RAGConfig.RRF_VECTOR_WEIGHT,
w_t = RAGConfig.RRF_TEXT_WEIGHT.

Channel routing:
- "body"     -> body vector / body fulltext indices
- "q_minus"  -> Q- (incoming) indices, used for grounded self-contained content
- "q_plus"   -> Q+ (outgoing) indices, used as expansion channel when Stage 1
                returns insufficient evidence (paper §3.2.3).
"""
import logging
from typing import Any

from core.config import RAGConfig


logger = logging.getLogger(__name__)


class HybridSearchMixin:
    @staticmethod
    def _channel_filter_clauses(channel: str) -> tuple[str, str]:
        if channel == "q_minus":
            return "node.q_minus_embedding IS NOT NULL", "node.q_minus_text IS NOT NULL"
        if channel == "q_plus":
            return "node.q_plus_embedding IS NOT NULL", "node.q_plus_text IS NOT NULL"
        return "", ""

    def _channel_index_names(self, channel: str) -> tuple[str, str]:
        if channel == "q_minus":
            return self.q_minus_vector_index, self.q_minus_text_index
        if channel == "q_plus":
            return self.q_plus_vector_index, self.q_plus_text_index
        return self.body_vector_index, self.body_text_index

    async def _hybrid_rrf_candidates(self, query: str, limit: int, channel: str = "body") -> list[dict[str, Any]]:
        embed = await self.llm.get_embedding(query)
        if not embed:
            logger.warning("Hybrid candidate collection aborted: empty query embedding.")
            return []

        vector_index, text_index = self._channel_index_names(channel)
        vector_filter, text_filter = self._channel_filter_clauses(channel)

        async with self.neo4j.driver.session() as session:
            query_vec = f"""
                CALL db.index.vector.queryNodes('{vector_index}', $limit, $embedding)
                YIELD node, score
                {('WHERE ' + vector_filter.strip()) if vector_filter.strip() else ''}
                RETURN node.id as id, node.title as title, node.sent_id as sent_id, node.page as page,
                       node.text as text, score, 'vector' as type, $channel as channel
            """
            vec_res = await session.run(query_vec, {  # type: ignore
                "limit": RAGConfig.VECTOR_SEARCH_LIMIT,
                "embedding": embed,
                "channel": channel,
            })
            vector_nodes = [dict(record) async for record in vec_res]

            safe_query = self._sanitize_fulltext_query(query)
            fulltext_query = safe_query or self._normalize_entity_term(query) or str(query or "")
            query_ft = f"""
                CALL db.index.fulltext.queryNodes('{text_index}', $query, {{limit: $limit}})
                YIELD node, score
                {('WHERE ' + text_filter.strip()) if text_filter.strip() else ''}
                RETURN node.id as id, node.title as title, node.sent_id as sent_id, node.page as page,
                       node.text as text, score, 'text' as type, $channel as channel
            """
            ft_res = await session.run(query_ft, {  # type: ignore
                "query": fulltext_query,
                "limit": RAGConfig.TEXT_SEARCH_LIMIT,
                "channel": channel,
            })
            text_nodes = [dict(record) async for record in ft_res]

        all_nodes: dict[str, dict[str, Any]] = {}

        def update_rrf(nodes: list[dict[str, Any]], weight: float = 1.0):
            for rank, node in enumerate(nodes):
                node_id = self._node_identity(node)
                if node_id not in all_nodes:
                    all_nodes[node_id] = dict(node)
                    all_nodes[node_id]["rrf_score"] = 0.0
                all_nodes[node_id]["rrf_score"] += weight * (1.0 / (RAGConfig.RRF_K_CONSTANT + rank))

        update_rrf(vector_nodes, weight=RAGConfig.RRF_VECTOR_WEIGHT)
        update_rrf(text_nodes, weight=RAGConfig.RRF_TEXT_WEIGHT)

        nodes = sorted(
            all_nodes.values(),
            key=lambda item: item.get("rrf_score", 0.0),
            reverse=True,
        )
        query_meta = self._extract_query_metadata(query)
        self._apply_retrieval_calibration(nodes, query_meta)
        return nodes[:limit]
