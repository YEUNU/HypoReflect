"""Rank-Based HOP Edge Pre-Construction (paper §3.1.4).

For each source chunk c_i with Q+ embedding q+_i:
1. Retrieve top-K_hop=10 candidates from the Q+ vector index by ANN
   (RAGConfig.HOP_LINK_LIMIT controls L_hop, the retained edge count).
2. Score each candidate c_j with a cross-encoder reranker on (Q+_i, c_j).
3. Keep edges where r >= tau_r=0.5 (RAGConfig.RERANKER_THRESHOLD), retaining
   the top L_hop edges by reranker score.

Multi-hop discovery happens once, at indexing time. The same tau_r is used at
retrieval (§3.2.3 graph traversal) so HOP edges follow the same scoring
criterion the system applies when reading them.
"""
import logging
from typing import Any

from core.config import RAGConfig


logger = logging.getLogger(__name__)


class HopEdgeMixin:
    async def _find_hop_candidates(self, hop_src: dict[str, Any]) -> list[dict[str, Any]]:
        if not hop_src.get("q_plus_embed"):
            return []

        query = f"""
            CALL db.index.vector.queryNodes($index, 15, $embed)
            YIELD node, score
            WHERE node.id <> $src_id
              AND node.source <> $src_source
              AND node.q_plus_embedding IS NOT NULL
            RETURN node.id as id, node.text as text, score
        """
        params = {
            "index": self.q_plus_vector_index,
            "embed": hop_src["q_plus_embed"],
            "src_id": hop_src["id"],
            "src_source": hop_src["source"],
        }
        results = await self.retry_query(query, params)
        if results:
            return results

        fallback_query = f"""
            CALL db.index.vector.queryNodes($index, 15, $embed)
            YIELD node, score
            WHERE node.id <> $src_id
              AND node.source <> $src_source
              AND node.q_minus_embedding IS NOT NULL
            RETURN node.id as id, node.text as text, score
        """
        fallback_params = {
            "index": self.q_minus_vector_index,
            "embed": hop_src["q_plus_embed"],
            "src_id": hop_src["id"],
            "src_source": hop_src["source"],
        }
        return await self.retry_query(fallback_query, fallback_params)

    async def _build_hop_edges_from_batch(self, current_batch: list[dict[str, Any]]) -> None:
        """Run rank-based HOP edge pre-construction over a flushed batch.

        Skipped when HOP_MODE != "offline" or Q+ ablation disables outgoing
        projections (Q+ is the sole HOP anchor)."""
        if (RAGConfig.HOP_MODE != "offline") or (not RAGConfig.ABLATION_Q_PLUS):
            logger.info(
                "Skipping offline HOP edge construction (HOP_MODE=%s, ABLATION_Q_PLUS=%s).",
                RAGConfig.HOP_MODE,
                RAGConfig.ABLATION_Q_PLUS,
            )
            return

        all_hop_edges = []
        for item in current_batch:
            batch_data = item["data"]
            hop_items = [batch_item for batch_item in batch_data if batch_item.get("q_plus_embed") is not None]

            for hop_src in hop_items:
                candidates = await self._find_hop_candidates(hop_src)
                if not candidates:
                    continue

                q_plus_text = " ".join(hop_src.get("q_plus", []))
                cand_texts = [candidate["text"] for candidate in candidates]

                try:
                    scores = await self.llm.rerank(q_plus_text, cand_texts, instruction=self._reranker_instruction())

                    valid_edges = []
                    for index, score in enumerate(scores):
                        if score >= RAGConfig.RERANKER_THRESHOLD:
                            valid_edges.append({
                                "src_id": hop_src["id"],
                                "tgt_id": candidates[index]["id"],
                                "score": score,
                            })

                    valid_edges = sorted(valid_edges, key=lambda item: item["score"], reverse=True)[:RAGConfig.HOP_LINK_LIMIT]
                    all_hop_edges.extend(valid_edges)
                except Exception as error:
                    logger.warning("Reranking for HOP edges failed: %s", error)

        if all_hop_edges:
            await self.retry_query(f"""
                UNWIND $edges AS edge
                MATCH (src:{self.chunk_label} {{id: edge.src_id}})
                MATCH (tgt:{self.chunk_label} {{id: edge.tgt_id}})
                MERGE (src)-[r:HOP]->(tgt)
                SET r.score = edge.score, r.type = 'pruned'
            """, {"edges": all_hop_edges})
