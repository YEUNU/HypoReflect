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
import asyncio
import logging
import os
from typing import Any

from core.config import RAGConfig


logger = logging.getLogger(__name__)


class HopEdgeMixin:
    async def _find_hop_candidates(self, hop_src: dict[str, Any]) -> list[dict[str, Any]]:
        if not hop_src.get("q_plus_embed"):
            return []

        # Same-company filter: in v14 we observed 18% of HOP edges crossed
        # company boundaries (e.g., AES → AMAZON, ADOBE → ACTIVISIONBLIZZARD)
        # because the cross-encoder reranker confused structurally similar
        # finance tables across unrelated tickers. FinanceBench queries are
        # company-anchored, so cross-company HOPs add retrieval noise without
        # answering the actual question. We restrict candidates to the same
        # company prefix; same-source is still excluded to keep edges
        # cross-document (paper §3.1.4 multi-hop discovery).
        src_company = hop_src.get("company") or ""

        query = f"""
            CALL db.index.vector.queryNodes($index, 15, $embed)
            YIELD node, score
            WHERE node.id <> $src_id
              AND node.source <> $src_source
              AND ($src_company = '' OR node.company = $src_company)
              AND node.q_plus_embedding IS NOT NULL
            RETURN node.id as id, node.text as text, score
        """
        params = {
            "index": self.q_plus_vector_index,
            "embed": hop_src["q_plus_embed"],
            "src_id": hop_src["id"],
            "src_source": hop_src["source"],
            "src_company": src_company,
        }
        results = await self.retry_query(query, params)
        if results:
            return results

        fallback_query = f"""
            CALL db.index.vector.queryNodes($index, 15, $embed)
            YIELD node, score
            WHERE node.id <> $src_id
              AND node.source <> $src_source
              AND ($src_company = '' OR node.company = $src_company)
              AND node.q_minus_embedding IS NOT NULL
            RETURN node.id as id, node.text as text, score
        """
        fallback_params = {
            "index": self.q_minus_vector_index,
            "embed": hop_src["q_plus_embed"],
            "src_id": hop_src["id"],
            "src_source": hop_src["source"],
            "src_company": src_company,
        }
        return await self.retry_query(fallback_query, fallback_params)

    async def _run_hop_pipeline(self, hop_items: list[dict[str, Any]]) -> int:
        """Parallel HOP edge construction over a flat list of hop_src dicts.

        Each hop_src must provide: id, source, company, q_plus_embed, q_plus.
        Runs ANN retrieval + cross-encoder rerank for each src concurrently
        (capped by RAG_HOP_RERANK_CONCURRENCY) and writes the resulting
        HOP edges to Neo4j in a single MERGE batch. Returns the number of
        edges written.

        The semaphore wraps the ENTIRE per-src op (ANN read + rerank).
        Wrapping only the rerank let tens of thousands of Neo4j ANN reads
        fire at once for large batches, exhausting the connection pool and
        silently dropping HOP MERGE writes.
        """
        rerank_sem = asyncio.Semaphore(
            max(1, int(os.environ.get("RAG_HOP_RERANK_CONCURRENCY", "64")))
        )
        reranker_instruction = self._reranker_instruction()

        async def _process_hop_src(hop_src: dict[str, Any]) -> list[dict[str, Any]]:
            async with rerank_sem:
                candidates = await self._find_hop_candidates(hop_src)
                if not candidates:
                    return []

                q_plus_text = " ".join(hop_src.get("q_plus", []))
                cand_texts = [candidate["text"] for candidate in candidates]

                try:
                    scores = await self.llm.rerank(
                        q_plus_text, cand_texts, instruction=reranker_instruction
                    )
                except Exception as error:
                    logger.warning("Reranking for HOP edges failed: %s", error)
                    return []

                valid_edges = []
                for index, score in enumerate(scores):
                    if score >= RAGConfig.RERANKER_THRESHOLD:
                        valid_edges.append({
                            "src_id": hop_src["id"],
                            "tgt_id": candidates[index]["id"],
                            "score": score,
                        })
                valid_edges.sort(key=lambda item: item["score"], reverse=True)
                return valid_edges[: RAGConfig.HOP_LINK_LIMIT]

        if not hop_items:
            return 0

        edge_groups = await asyncio.gather(
            *[_process_hop_src(src) for src in hop_items],
            return_exceptions=False,
        )
        all_hop_edges = [edge for group in edge_groups for edge in group]

        if all_hop_edges:
            await self.retry_query(f"""
                UNWIND $edges AS edge
                MATCH (src:{self.chunk_label} {{id: edge.src_id}})
                MATCH (tgt:{self.chunk_label} {{id: edge.tgt_id}})
                MERGE (src)-[r:HOP]->(tgt)
                SET r.score = edge.score, r.type = 'pruned'
            """, {"edges": all_hop_edges})
        return len(all_hop_edges)

    async def build_all_hop_edges(self) -> None:
        """One-shot HOP edge pre-construction over the COMPLETE graph
        (paper §3.1.4: 'Multi-hop discovery happens once, at indexing time').

        Replaces the prior per-batch HOP construction in
        _flush_graph_batch_unlocked, which produced an asymmetric graph:
        chunks written in the first NEO4J_BATCH_SIZE=25 docs only saw 24
        other docs as candidates, while late-batch chunks saw the entire
        corpus. Running once at the end gives every chunk the same
        candidate pool — the symmetric, paper-aligned behavior.
        """
        if (RAGConfig.HOP_MODE != "offline") or (not RAGConfig.ABLATION_Q_PLUS):
            logger.info(
                "Skipping offline HOP edge construction (HOP_MODE=%s, ABLATION_Q_PLUS=%s).",
                RAGConfig.HOP_MODE,
                RAGConfig.ABLATION_Q_PLUS,
            )
            return

        rows = await self.retry_query(f"""
            MATCH (c:{self.chunk_label})
            WHERE c.q_plus_embedding IS NOT NULL
              AND c.q_plus_text IS NOT NULL AND c.q_plus_text <> ''
            RETURN c.id AS id, c.source AS source, c.company AS company,
                   c.q_plus_embedding AS q_plus_embed, c.q_plus_text AS q_plus_text
        """)
        if not rows:
            logger.info("build_all_hop_edges: no Q+ chunks; skipping.")
            return

        hop_items = [
            {
                "id": r["id"],
                "source": r["source"],
                "company": r.get("company") or "",
                "q_plus_embed": r["q_plus_embed"],
                "q_plus": [r["q_plus_text"]],  # already concatenated at write-time
            }
            for r in rows
        ]
        logger.info("build_all_hop_edges: scoring %d Q+ chunks for HOP edges...", len(hop_items))
        n_edges = await self._run_hop_pipeline(hop_items)
        logger.info("build_all_hop_edges: wrote %d HOP edges.", n_edges)
