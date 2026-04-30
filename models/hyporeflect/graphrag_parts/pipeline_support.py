import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

import numpy as np

from core.config import RAGConfig
from utils.prompts import (
    GROUP_SUMMARY_PROMPT,
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
    PAGE_SUMMARY_PROMPT,
)


logger = logging.getLogger(__name__)


def _cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def _make_semantic_chunk_id(source, title, sent_id):
    content_sig = f"{source}-{title}-{sent_id}"
    return hashlib.md5(content_sig.encode()).hexdigest()


class PipelineSupport:
    def _save_debug(self, doc_name: str, step: str, data: Any):
        doc_dir = os.path.join(self.debug_output_dir, doc_name.replace(" ", "_").replace("/", "_"))
        os.makedirs(doc_dir, exist_ok=True)
        filepath = os.path.join(doc_dir, f"{step}.json")
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        logger.info("[DEBUG] Saved %s to %s", step, filepath)

    async def extract_knowledge(self, content: str, source: str = "") -> dict[str, Any]:
        lines = content.split("\n")
        title = "Unknown"
        if lines and lines[0].startswith("Document: "):
            title = lines[0].replace("Document: ", "").strip()
        elif lines and lines[0].startswith("Title: "):
            title = lines[0].replace("Title: ", "").strip()

        logger.info("[%s] Content head (500 chars): %r", title, content[:500])

        page_pattern = re.compile(r"-+\s*Page\s*(\d+)\s*-+", re.IGNORECASE)
        matches = list(page_pattern.finditer(content))

        pages = []
        if matches:
            for index, start_match in enumerate(matches):
                page_num = int(start_match.group(1))
                content_start = start_match.end()
                if index < len(matches) - 1:
                    content_end = matches[index + 1].start()
                else:
                    content_end = len(content)

                page_text = content[content_start:content_end].strip()
                if page_text:
                    pages.append({"num": page_num, "content": page_text})

        logger.info("[%s] Parsed %d pages from content.", title, len(pages))

        if not pages:
            logger.info("[%s] No page markers found, using standard chunking fallback", title)
            lines = content.split("\n")
            start_idx = 0
            if lines and (lines[0].startswith("Title: ") or lines[0].startswith("Document: ")):
                start_idx = 1
            content_body = "\n".join(lines[start_idx:])
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content_body) if s.strip()]
            pages = [{"num": 1, "content": " ".join(sentences)}]
            if not sentences:
                return {"title": title, "chunks": []}

        if not RAGConfig.ABLATION_ADAPTIVE_CHUNKING:
            logger.info("Ablation: Using fixed page-based grouping (no adaptive similarity).")
            page_groups = []
            page_summaries = [page["content"][:200] for page in pages]
            for index, page in enumerate(pages):
                page_groups.append({
                    "pages": [page["num"]],
                    "content": page["content"],
                    "start_page": page["num"],
                    "page_summaries": [page_summaries[index]],
                    "group_summary": "",
                })
        else:
            async def get_page_summary(page_text: str) -> str:
                prompt = PAGE_SUMMARY_PROMPT.format(text=page_text[:2000])
                messages = [{"role": "user", "content": prompt}]
                try:
                    return await self.indexing_llm.generate_response(messages, apply_default_sampling=False)
                except Exception:
                    sentences = re.split(r"(?<=[.!?])\s+", page_text.strip())
                    return " ".join(sentences[:2]) if sentences else page_text[:200]

            page_summary_tasks = [get_page_summary(page["content"]) for page in pages]
            page_summaries = await asyncio.gather(*page_summary_tasks)
            logger.info("[%s] Generated %d page summaries via LLM", title, len(page_summaries))

            page_embeds = await self.llm.get_embeddings(page_summaries)
            page_groups = []
            current_group_start = 0
            page_similarity_threshold = RAGConfig.PAGE_SIMILARITY_THRESHOLD

            for index in range(len(pages)):
                should_split = False
                if index < len(pages) - 1:
                    similarity = _cosine_similarity(page_embeds[index], page_embeds[index + 1])
                    if similarity < page_similarity_threshold:
                        should_split = True
                else:
                    should_split = True

                if should_split:
                    group_pages = pages[current_group_start:index + 1]
                    group_content = "\n\n".join([page["content"] for page in group_pages])
                    group_page_range = [page["num"] for page in group_pages]
                    group_page_summaries = [page_summaries[j] for j in range(current_group_start, index + 1)]
                    page_groups.append({
                        "pages": group_page_range,
                        "content": group_content,
                        "start_page": group_page_range[0],
                        "page_summaries": group_page_summaries,
                    })
                    current_group_start = index + 1

            logger.info("[%s] Grouped %d pages into %d semantic groups", title, len(pages), len(page_groups))

            async def summarize_group(group):
                page_summaries_text = "\n".join([f"- {summary}" for summary in group["page_summaries"]])
                prompt = GROUP_SUMMARY_PROMPT.format(page_summaries=page_summaries_text)
                messages = [{"role": "user", "content": prompt}]
                return await self.indexing_llm.generate_response(messages, apply_default_sampling=False)

            group_summary_tasks = [summarize_group(group) for group in page_groups]
            group_summaries = await asyncio.gather(*group_summary_tasks)

            for index, group in enumerate(page_groups):
                group["group_summary"] = group_summaries[index].strip() if group_summaries[index] else ""

            logger.info("[%s] Generated %d group summaries", title, len(group_summaries))

        self._save_debug(title, "step1_page_summaries", [
            {"page": page["num"], "summary": summary}
            for page, summary in zip(pages, page_summaries)
        ])
        self._save_debug(title, "step2_page_groups", [
            {
                "group_idx": index,
                "pages": group["pages"],
                "start_page": group["start_page"],
                "page_summaries": group["page_summaries"],
                "group_summary": group["group_summary"],
            }
            for index, group in enumerate(page_groups)
        ])

        final_chunks = []
        global_sent_id = 0
        first_group_summary = page_groups[0].get("group_summary", "") if page_groups else ""
        intro_summary = first_group_summary if first_group_summary else f"Document: {title}"
        milestone_summaries = []
        recent_summary = ""
        milestone_interval = RAGConfig.MILESTONE_INTERVAL
        chunk_sem = asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_LLM_CALLS)

        async def process_group(pg_idx, page_group):
            nonlocal global_sent_id, recent_summary
            group_content = page_group["content"]
            group_start_page = page_group["start_page"]
            if not group_content:
                return []

            raw_sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", group_content) if sentence.strip()]
            if not raw_sentences:
                return []

            processed_sentences = []
            table_buffer = []
            for line in raw_sentences:
                if "|" in line:
                    table_buffer.append(line)
                else:
                    if table_buffer:
                        processed_sentences.extend(await self._table_to_text(table_buffer))
                        table_buffer = []
                    processed_sentences.append(line)
            if table_buffer:
                processed_sentences.extend(await self._table_to_text(table_buffer))

            group_chunks = []
            sent_embeds = await self.llm.get_embeddings(processed_sentences)

            if len(sent_embeds) > 1:
                similarities = [
                    _cosine_similarity(sent_embeds[k], sent_embeds[k + 1])
                    for k in range(len(sent_embeds) - 1)
                ]
                avg_sim = sum(similarities) / len(similarities)
                adaptive_threshold = min(RAGConfig.SIMILARITY_THRESHOLD, avg_sim - 0.1)
            else:
                adaptive_threshold = RAGConfig.SIMILARITY_THRESHOLD

            min_chunk_sentences = RAGConfig.MIN_CHUNK_SENTENCES
            current_group = []

            for index in range(len(processed_sentences)):
                current_group.append(processed_sentences[index])
                should_split = False
                if index < len(processed_sentences) - 1:
                    if len(current_group) >= min_chunk_sentences:
                        similarity = _cosine_similarity(sent_embeds[index], sent_embeds[index + 1])
                        if similarity < adaptive_threshold:
                            should_split = True
                else:
                    should_split = True

                if should_split:
                    chunk_text = " ".join(current_group)

                    async with chunk_sem:
                        if not RAGConfig.ABLATION_ROLLING_SUMMARY:
                            rolling_context = f"Document: {title}"
                        else:
                            context_parts = [intro_summary]
                            if milestone_summaries:
                                context_parts.append(f"Key points: {' | '.join(milestone_summaries[-2:])}")
                            if recent_summary:
                                context_parts.append(f"Recent: {recent_summary}")
                            rolling_context = " || ".join(context_parts)

                        q_data = await self.extract_hoprag_queries_with_rolling(chunk_text, title, rolling_context)

                    group_chunks.append({
                        "page": group_start_page,
                        "text": chunk_text,
                        "title": title,
                        "q_minus": q_data.get("q_minus", []),
                        "q_plus": q_data.get("q_plus", []),
                        "summary": q_data.get("summary", ""),
                    })

                    new_summary = q_data.get("summary", "")
                    if new_summary:
                        recent_summary = new_summary

                    current_group = []

            return group_chunks

        for index, page_group in enumerate(page_groups):
            logger.info(
                "[%s] Processing Page Group %d/%d (Pages: %s)",
                title,
                index + 1,
                len(page_groups),
                page_group["pages"],
            )
            group_chunks = await process_group(index, page_group)
            for group_chunk in group_chunks:
                group_chunk["sent_id"] = global_sent_id
                final_chunks.append(group_chunk)
                global_sent_id += 1

                if global_sent_id % milestone_interval == 0 and group_chunk["summary"]:
                    milestone_summaries.append(group_chunk["summary"])

        self._save_debug(title, "step3_final_chunks", [
            {
                "sent_id": chunk["sent_id"],
                "page": chunk["page"],
                "text": chunk["text"][:200] + "...",
                "q_minus": chunk["q_minus"],
                "q_plus": chunk["q_plus"],
                "summary": chunk["summary"],
            }
            for chunk in final_chunks
        ])

        return {"title": title, "chunks": final_chunks}

    async def extract_hoprag_queries_with_rolling(self, chunk: str, title: str, running_summary: str) -> dict[str, Any]:
        prompt_template = HOPRAG_PROMPT
        text_prompt = prompt_template.format(
            chunk=chunk,
            global_context=f"Document: {title}. Previous context: {running_summary}",
        )
        messages = [{"role": "user", "content": text_prompt}, {"role": "user", "content": HOPRAG_FORMAT_INSTRUCTION}]
        try:
            data = await self.indexing_llm.generate_json(messages, apply_default_sampling=False)
            return {
                "q_minus": data.get("q_minus", []),
                "q_plus": data.get("q_plus", []),
                "summary": data.get("summary", ""),
            }
        except Exception as error:
            logger.error("HopRAG extraction failed: %s", error)
            return {"q_minus": [], "q_plus": [], "summary": ""}

    async def build_graph(self, knowledge: dict[str, Any], source: str, document_filename: str):
        await self._ensure_index_ready()

        chunks = knowledge.get("chunks", [])
        if not chunks:
            return

        body_texts = [str(chunk.get("text", "") or "") for chunk in chunks]
        q_minus_texts = [
            " ".join(self._dedupe_preserve_order([str(value or "") for value in chunk.get("q_minus", [])])).strip()
            for chunk in chunks
        ]

        gated_q_plus_per_chunk: list[list[str]] = []
        q_plus_texts: list[str] = []
        for chunk in chunks:
            raw_q_plus = self._dedupe_preserve_order([str(value or "") for value in chunk.get("q_plus", [])])
            gated_q_plus = [
                question for question in raw_q_plus
                if self._is_high_quality_q_plus(
                    question,
                    str(chunk.get("title", "") or ""),
                    str(chunk.get("text", "") or ""),
                )
            ]
            gated_q_plus_per_chunk.append(gated_q_plus)
            q_plus_texts.append(" ".join(gated_q_plus).strip())

        body_embeds, q_minus_embeds, q_plus_embeds = await asyncio.gather(
            self._embed_sparse_texts(body_texts),
            self._embed_sparse_texts(q_minus_texts),
            self._embed_sparse_texts(q_plus_texts),
        )

        batch_data = []
        for index, chunk in enumerate(chunks):
            body_embedding = body_embeds[index] if index < len(body_embeds) else []
            q_minus_embedding = q_minus_embeds[index] if index < len(q_minus_embeds) else []
            q_plus_embedding = q_plus_embeds[index] if index < len(q_plus_embeds) else []
            q_plus_items = gated_q_plus_per_chunk[index] if index < len(gated_q_plus_per_chunk) else []

            primary_embedding = q_minus_embedding if q_minus_embedding else body_embedding
            if not primary_embedding:
                logger.warning(
                    "Skipping chunk with missing embedding: source=%s title=%s sent_id=%s",
                    source,
                    chunk.get("title", ""),
                    chunk.get("sent_id", -1),
                )
                continue
            chunk_id = _make_semantic_chunk_id(source, chunk["title"], chunk["sent_id"])
            batch_data.append({
                "id": chunk_id,
                "text": chunk["text"],
                "source": source,
                "title": chunk["title"],
                "sent_id": chunk["sent_id"],
                "page": chunk.get("page", 0),
                "embedding": primary_embedding,
                "body_embedding": body_embedding if body_embedding else None,
                "q_minus_embedding": q_minus_embedding if q_minus_embedding else None,
                "q_plus_embedding": q_plus_embedding if q_plus_embedding and q_plus_items else None,
                "q_minus_text": q_minus_texts[index] if index < len(q_minus_texts) else "",
                "q_plus_text": q_plus_texts[index] if index < len(q_plus_texts) else "",
                "q_plus": q_plus_items,
                "q_plus_embed": q_plus_embedding if q_plus_embedding and q_plus_items else None,
                "chunk_summary": chunk["summary"],
            })

        if not batch_data:
            logger.warning("All chunks skipped for %s due to missing embeddings.", source)
            return

        async with self._batch_lock:
            self._pending_batch.append({"data": batch_data, "doc_id": document_filename})
            if len(self._pending_batch) >= RAGConfig.NEO4J_BATCH_SIZE:
                await self._flush_graph_batch_unlocked()

    async def flush_graph_batch(self):
        async with self._batch_lock:
            await self._flush_graph_batch_unlocked()

    async def _find_hop_candidates(self, hop_src: dict[str, Any]) -> list[dict[str, Any]]:
        if not hop_src.get("q_plus_embed"):
            return []

        query = f"""
            CALL db.index.vector.queryNodes($index, 10, $embed)
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
            CALL db.index.vector.queryNodes($index, 10, $embed)
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

    async def _flush_graph_batch_unlocked(self):
        if not self._pending_batch:
            return

        current_batch = self._pending_batch
        self._pending_batch = []

        for item in current_batch:
            await self.retry_query(f"""
                MATCH (d:{self.doc_label} {{filename: $doc_id}})
                WITH d
                UNWIND $batch AS item
                MERGE (c:{self.chunk_label} {{id: item.id}})
                SET c.text = item.text, c.source = item.source, c.title = item.title,
                    c.sent_id = item.sent_id, c.page = item.page, c.corpus = $corpus,
                    c.embedding = item.embedding,
                    c.body_embedding = item.body_embedding, c.q_minus_embedding = item.q_minus_embedding,
                    c.q_plus_embedding = item.q_plus_embedding, c.q_minus_text = item.q_minus_text,
                    c.q_plus_text = item.q_plus_text, c.chunk_summary = item.chunk_summary
                MERGE (d)-[:CONTAINS]->(c)
            """, {"batch": item["data"], "doc_id": item["doc_id"], "corpus": self.corpus_tag})

            await self.retry_query(f"""
                UNWIND range(0, size($batch)-2) AS i
                MATCH (c1:{self.chunk_label} {{id: $batch[i].id}})
                MATCH (c2:{self.chunk_label} {{id: $batch[i+1].id}})
                MERGE (c1)-[:NEXT]->(c2)
            """, {"batch": item["data"]})

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

    async def _build_graph_tx(self, tx, batch, doc_id):
        pass
