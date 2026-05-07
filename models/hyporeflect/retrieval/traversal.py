"""Graph traversal over pre-built NEXT/HOP edges (paper §3.2.3 "Graph Traversal").

Starting from reranked seed nodes, the system expands along NEXT (sequential)
and HOP (semantic, pre-built §3.1.4) edges. At each hop a continuation
decision asks the LLM whether the accumulated context is sufficient; if yes
the traversal stops early.

Two HOP modes:
- "offline" (paper canonical): traverse the pre-built [:NEXT|HOP] edges
- "runtime" (HopRAG-style fallback): follow only [:NEXT] in the graph but
  expand the frontier at query time via Q+ ANN + cross-encoder rerank.
"""
import logging
from typing import Any

from core.config import RAGConfig
from utils.prompts import SEARCH_CONTINUATION_FORMAT_INSTRUCTION


logger = logging.getLogger(__name__)


class TraversalMixin:
    async def graph_search(self, entities: list[str], depth: int = 2, top_k: int = 5) -> tuple:
        normalized_entities: list[str] = []
        for entity in entities:
            normalized = self._normalize_entity_term(entity)
            if normalized:
                normalized_entities.append(normalized)
        seed_query = " ".join(normalized_entities).strip() or " ".join(entities).strip()
        if not seed_query:
            return "", []

        try:
            depth = max(1, min(int(depth), 4))
        except Exception:
            depth = 4

        seed_top_k = max(1, min(max(1, top_k - 1), RAGConfig.GRAPH_SEARCH_LIMIT))
        _, seed_nodes = await self.retrieve(seed_query, top_k=seed_top_k)
        seed_ids = [
            str(node.get("id")).strip()
            for node in seed_nodes
            if node.get("id") is not None and str(node.get("id")).strip()
        ]

        if not seed_ids:
            return await self.retrieve(seed_query, top_k=top_k)
        search_query = " ".join(entities).strip() or " ".join(normalized_entities)
        search_query_meta = self._extract_query_metadata(search_query)
        step_limit = max(RAGConfig.GRAPH_SEARCH_LIMIT, top_k * 6)
        frontier_ids = [seed_id for seed_id in seed_ids if seed_id]
        visited_ids = set(frontier_ids)
        collected: dict[str, dict[str, Any]] = {}

        async def _rerank_and_gate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not candidates:
                return []
            texts = [candidate.get("text", "") for candidate in candidates]
            try:
                scores = await self.llm.rerank(search_query, texts, instruction=self._reranker_instruction())
            except Exception as error:
                logger.warning("Graph search reranking failed: %s", error)
                scores = [0.0] * len(candidates)

            self._apply_retrieval_calibration(candidates, search_query_meta)
            for index, score in enumerate(scores):
                candidates[index]["rerank_score"] = score
                candidates[index]["final_score"] = (
                    score
                    + (RAGConfig.META_BOOST_WEIGHT * candidates[index].get("meta_boost", 0.0))
                    - (RAGConfig.BOILERPLATE_PENALTY_WEIGHT * candidates[index].get("boilerplate_penalty", 0.0))
                )

            reranked = sorted(candidates, key=lambda item: item.get("final_score", 0.0), reverse=True)
            company_matched = [node for node in reranked if self._node_matches_company(node, search_query_meta)]
            if company_matched and any((search_query_meta.get("company_keys") or [])):
                reranked = company_matched + [node for node in reranked if not self._node_matches_company(node, search_query_meta)]

            gated = [node for node in reranked if node.get("rerank_score", 0.0) >= RAGConfig.RERANKER_THRESHOLD]
            if gated:
                return gated
            return reranked[: max(top_k, 3)]

        async def _need_more_for_next_depth(nodes_for_judge: list[dict[str, Any]]) -> bool:
            if not nodes_for_judge:
                return True
            ranked = sorted(
                nodes_for_judge,
                key=lambda item: item.get("rerank_score", 0.0),
                reverse=True,
            )[: max(top_k, 6)]
            context_preview = "\n\n".join([
                f"[[{node.get('title', 'Unknown')}, Page {node.get('page', 0)}, Chunk {node.get('sent_id', -1)}]]\n"
                f"{str(node.get('text', '') or '')[:450]}"
                for node in ranked
            ])
            messages = [
                {"role": "user", "content": self._search_continuation_prompt().format(query=search_query, context=context_preview)},
                {"role": "user", "content": SEARCH_CONTINUATION_FORMAT_INSTRUCTION},
            ]
            try:
                decision_data = await self.llm.generate_json(messages, apply_default_sampling=False)
                decision = str((decision_data or {}).get("decision", "INSUFFICIENT")).strip().upper()
                need_more = decision != "SUFFICIENT"
                logger.info("Graph depth continuation decision=%s (need_more=%s)", decision, need_more)
                return need_more
            except Exception as error:
                logger.warning("Graph continuation check failed; continuing depth expansion: %s", error)
                return True

        seed_gated = await _rerank_and_gate(seed_nodes)
        frontier_ids = []
        for node in seed_gated:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            frontier_ids.append(node_id)
            previous = collected.get(node_id)
            if (previous is None) or (
                node.get("final_score", node.get("rerank_score", 0.0))
                > previous.get("final_score", previous.get("rerank_score", 0.0))
            ):
                collected[node_id] = node

        if collected:
            need_more = await _need_more_for_next_depth(list(collected.values()))
            if not need_more:
                nodes = sorted(
                    collected.values(),
                    key=lambda item: item.get("final_score", item.get("rerank_score", 0.0)),
                    reverse=True,
                )[:top_k]
                return self._build_context_from_nodes(nodes), nodes

        for hop_index in range(depth):
            if not frontier_ids:
                break

            edge_pattern = "[:NEXT]" if RAGConfig.HOP_MODE == "runtime" else "[:NEXT|HOP]"
            async with self.neo4j.driver.session() as session:
                query = f"""
                    UNWIND $frontier_ids AS src_id
                    MATCH (src:{self.chunk_label} {{id: src_id}})-{edge_pattern}->(related:{self.chunk_label})
                    WHERE NOT related.id IN $visited_ids
                    RETURN DISTINCT related.id as id, related.title as title, related.sent_id as sent_id,
                                    related.page as page, related.text as text, related.source as source
                    LIMIT $limit
                """
                result = await session.run(query, {  # type: ignore
                    "frontier_ids": frontier_ids,
                    "visited_ids": list(visited_ids),
                    "limit": step_limit,
                })
                candidates = [dict(record) async for record in result]

            if RAGConfig.HOP_MODE == "runtime":
                runtime_cands = await self._runtime_hop_candidates(
                    frontier_ids=frontier_ids,
                    visited_ids=visited_ids,
                    step_limit=step_limit,
                )
                seen_ids = {str(node.get("id", "")) for node in candidates}
                for cand in runtime_cands:
                    cand_id = str(cand.get("id", ""))
                    if cand_id and cand_id not in seen_ids:
                        candidates.append(cand)
                        seen_ids.add(cand_id)

            if not candidates:
                break

            gated_nodes = await _rerank_and_gate(candidates)
            if not gated_nodes:
                break

            next_frontier: list[str] = []
            for node in gated_nodes:
                node_id = str(node.get("id", "")).strip()
                if not node_id:
                    continue
                visited_ids.add(node_id)
                next_frontier.append(node_id)
                previous = collected.get(node_id)
                if (previous is None) or (
                    node.get("final_score", node.get("rerank_score", 0.0))
                    > previous.get("final_score", previous.get("rerank_score", 0.0))
                ):
                    collected[node_id] = node

            frontier_ids = next_frontier[:step_limit]
            if hop_index < depth - 1 and collected:
                need_more = await _need_more_for_next_depth(list(collected.values()))
                if not need_more:
                    break

        nodes = sorted(
            collected.values(),
            key=lambda item: item.get("final_score", item.get("rerank_score", 0.0)),
            reverse=True,
        )[:top_k]
        if not nodes:
            return "", []
        return self._build_context_from_nodes(nodes), nodes

    async def _runtime_hop_candidates(
        self,
        frontier_ids: list[str],
        visited_ids: set,
        step_limit: int,
    ) -> list[dict[str, Any]]:
        """Runtime mirror of offline HOP construction: for each frontier node,
        ANN-search the q_plus index with the node's own q_plus embedding.

        This is the dynamic (HopRAG-style) counterpart to pre-built HOP edges.
        Used only when RAGConfig.HOP_MODE == "runtime".
        """

        if not frontier_ids:
            return []

        async with self.neo4j.driver.session() as session:
            fetch_query = f"""
                UNWIND $frontier_ids AS fid
                MATCH (n:{self.chunk_label} {{id: fid}})
                WHERE n.q_plus_embedding IS NOT NULL
                RETURN n.id as src_id, n.q_plus_embedding as embed, n.source as src_source
            """
            fetch_res = await session.run(fetch_query, {"frontier_ids": frontier_ids})  # type: ignore
            sources = [dict(record) async for record in fetch_res]

        if not sources:
            return []

        per_source_k = max(1, min(10, step_limit))
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for source in sources:
            ann_query = f"""
                CALL db.index.vector.queryNodes('{self.q_plus_vector_index}', $k, $embed)
                YIELD node, score
                WHERE node.id <> $src_id
                  AND node.source <> $src_source
                  AND NOT node.id IN $visited
                RETURN node.id as id, node.title as title, node.sent_id as sent_id,
                       node.page as page, node.text as text, node.source as source, score
            """
            try:
                rows = await self.retry_query(
                    ann_query,
                    {
                        "k": per_source_k,
                        "embed": source["embed"],
                        "src_id": source["src_id"],
                        "src_source": source["src_source"],
                        "visited": list(visited_ids),
                    },
                )
            except Exception as error:
                logger.warning("Runtime HOP ANN query failed: %s", error)
                continue
            for row in rows or []:
                cid = str(row.get("id") or "")
                if not cid or cid in candidates_by_id:
                    continue
                candidates_by_id[cid] = dict(row)

        ranked = sorted(
            candidates_by_id.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        return ranked[:step_limit]
