import logging
from typing import Any

from core.config import RAGConfig
from utils.prompts import SEARCH_CONTINUATION_FORMAT_INSTRUCTION


logger = logging.getLogger(__name__)


class RetrievalSupport:
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

            async with self.neo4j.driver.session() as session:
                query = f"""
                    UNWIND $frontier_ids AS src_id
                    MATCH (src:{self.chunk_label} {{id: src_id}})-[:NEXT|HOP]->(related:{self.chunk_label})
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

        for index, query_text in enumerate(query_variants):
            query_weight = 1.0 if index == 0 else RAGConfig.QUERY_REWRITE_WEIGHT
            q_minus_nodes = await self._hybrid_rrf_candidates(query_text, limit=candidate_limit_per_query, channel="q_minus")
            body_nodes = await self._hybrid_rrf_candidates(query_text, limit=max(10, top_k * 4), channel="body")
            _accumulate(stage1_merged, q_minus_nodes, "stage1_rrf_score", query_weight * 0.7)
            _accumulate(stage1_merged, body_nodes, "stage1_rrf_score", query_weight * 0.3)

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

        if not need_expand:
            for node in stage1_nodes:
                node.pop("stage1_rrf_score", None)
                node.pop("stage2_rrf_score", None)
                node.pop("stage2_support_score", None)
            return self._build_context_from_nodes(stage1_nodes), stage1_nodes

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
