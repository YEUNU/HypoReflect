"""Neo4j storage for the indexing pipeline.

Owns the graph-write side: index lifecycle (vector + fulltext), document and
chunk MERGE, NEXT edge creation, and batched writes. HOP edges are delegated
to HopEdgeMixin (paper §3.1.4); document-level summaries use indexing_llm.

Three node types per (strategy, corpus_tag) namespace:
- Body / Q- / Q+ vector indices, plus matching BM25-style fulltext indices.
"""
import asyncio
import logging
import random
import re
from typing import Any, Optional

from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from core.config import RAGConfig
from utils.prompts import (
    GLOBAL_SUMMARY_FORMAT_INSTRUCTION,
    GLOBAL_SUMMARY_PROMPT,
)

from .chunking import _make_semantic_chunk_id


logger = logging.getLogger(__name__)


class GraphWriterMixin:
    async def setup_index(self):
        try:
            analyzer = re.sub(r"[^a-zA-Z0-9_\-]", "", RAGConfig.FULLTEXT_ANALYZER) or "english"
            vector_specs = [
                (self.body_vector_index, "embedding"),
                (self.q_minus_vector_index, "q_minus_embedding"),
                (self.q_plus_vector_index, "q_plus_embedding"),
            ]
            for index_name, property_name in vector_specs:
                await self.neo4j.execute_query(
                    f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{self.chunk_label}) ON (n.{property_name})
                    OPTIONS {{indexConfig: {{`vector.dimensions`: $dimensions, `vector.similarity_function`: 'cosine'}}}} """
                    ,
                    {"dimensions": self.vector_dimensions},
                )

            if RAGConfig.RECREATE_TEXT_INDEX:
                for index_name in [
                    self.body_text_index,
                    self.q_minus_text_index,
                    self.q_plus_text_index,
                ]:
                    await self.neo4j.execute_query(f"DROP INDEX {index_name} IF EXISTS")

            await self.neo4j.execute_query(f"""
                CREATE FULLTEXT INDEX {self.body_text_index} IF NOT EXISTS
                FOR (n:{self.chunk_label}) ON EACH [n.text, n.chunk_summary]
                OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
            await self.neo4j.execute_query(f"""
                CREATE FULLTEXT INDEX {self.q_minus_text_index} IF NOT EXISTS
                FOR (n:{self.chunk_label}) ON EACH [n.q_minus_text]
                OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)
            await self.neo4j.execute_query(f"""
                CREATE FULLTEXT INDEX {self.q_plus_text_index} IF NOT EXISTS
                FOR (n:{self.chunk_label}) ON EACH [n.q_plus_text]
                OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}} """)

            await self.neo4j.execute_query(
                f"CREATE INDEX {self.chunk_label}_id_idx IF NOT EXISTS FOR (n:{self.chunk_label}) ON (n.id)")
            await self.neo4j.execute_query(
                f"CREATE INDEX {self.doc_label}_fn_idx IF NOT EXISTS FOR (n:{self.doc_label}) ON (n.filename)")
        except Exception as error:
            logger.error("Index creation error: %s", error)

    async def _ensure_index_ready(self):
        if self._index_ready:
            return
        async with self._index_setup_lock:
            if self._index_ready:
                return
            await self.setup_index()
            self._index_ready = True

    @staticmethod
    def _is_retryable_neo4j_error(error: Exception) -> bool:
        if isinstance(error, (TransientError, ServiceUnavailable, SessionExpired)):
            return True
        code = str(getattr(error, "code", "") or "")
        text = str(error)
        markers = [
            "DeadlockDetected",
            "Neo.TransientError",
            "TransientError",
            "ServiceUnavailable",
        ]
        return any(marker in code or marker in text for marker in markers)

    async def retry_query(self, query: str, parameters: Optional[dict[str, Any]] = None):
        for attempt in range(self.max_retries):
            try:
                return await self.neo4j.execute_query(query, parameters)
            except Exception as error:
                if not self._is_retryable_neo4j_error(error):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                delay = (RAGConfig.RETRY_DELAY * (2 ** attempt)) + random.uniform(0, RAGConfig.RETRY_DELAY)
                logger.warning(
                    "Neo4j transient error (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    error,
                )
                await asyncio.sleep(delay)

    async def create_document_node(self, filename: str, metadata: dict[str, Any]) -> str:
        query = f"""
            MERGE (d:{self.doc_label} {{filename: $filename}})
            SET d.corpus = $corpus, d.title = $title, d.updated_at = timestamp()
            RETURN d.filename as id
        """
        async with self._batch_lock:
            results = await self.retry_query(query, {
                "filename": filename,
                "title": metadata.get("title", filename),
                "corpus": self.corpus_tag
            })
        return results[0]["id"] if results else filename

    async def summarize_document(self, filename: str):
        async def _get_chunks():
            async with self.neo4j.driver.session() as session:
                query = f"""
                    MATCH (d:{self.doc_label} {{filename: $filename}})-[:CONTAINS]->(c:{self.chunk_label})
                    RETURN c.text as text ORDER BY c.sent_id ASC LIMIT $limit
                """
                result = await session.run(query, {  # type: ignore
                    "filename": filename,
                    "limit": RAGConfig.CONTEXT_FETCH_LIMIT
                })
                return [record["text"] async for record in result]

        chunks = await _get_chunks()
        if not chunks:
            return
        context_text = "\n\n".join(chunks)
        prompt = GLOBAL_SUMMARY_PROMPT.format(text=context_text)
        messages = [{"role": "user", "content": prompt}, {"role": "user", "content": GLOBAL_SUMMARY_FORMAT_INSTRUCTION}]

        try:
            summary_data = await self.indexing_llm.generate_json(messages, apply_default_sampling=False)
            summary_text = summary_data.get("summary", "No summary.")
            await self.retry_query(
                f"MATCH (d:{self.doc_label} {{filename: $filename}}) SET d.summary = $summary",
                {"filename": filename, "summary": summary_text}
            )
        except Exception as error:
            logger.error("Summarize failed for %s: %s", filename, error)

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

        empty_per_chunk = [[] for _ in body_texts]
        embed_jobs = [self._embed_sparse_texts(body_texts)]
        embed_jobs.append(
            self._embed_sparse_texts(q_minus_texts) if RAGConfig.ABLATION_Q_MINUS
            else asyncio.sleep(0, result=empty_per_chunk)
        )
        embed_jobs.append(
            self._embed_sparse_texts(q_plus_texts) if RAGConfig.ABLATION_Q_PLUS
            else asyncio.sleep(0, result=empty_per_chunk)
        )
        body_embeds, q_minus_embeds, q_plus_embeds = await asyncio.gather(*embed_jobs)

        batch_data = []
        for index, chunk in enumerate(chunks):
            body_embedding = body_embeds[index] if index < len(body_embeds) else []
            q_minus_embedding = (
                q_minus_embeds[index] if RAGConfig.ABLATION_Q_MINUS and index < len(q_minus_embeds) else []
            )
            q_plus_embedding = (
                q_plus_embeds[index] if RAGConfig.ABLATION_Q_PLUS and index < len(q_plus_embeds) else []
            )
            q_plus_items = (
                gated_q_plus_per_chunk[index]
                if RAGConfig.ABLATION_Q_PLUS and index < len(gated_q_plus_per_chunk)
                else []
            )

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
                "q_minus_text": (
                    q_minus_texts[index] if RAGConfig.ABLATION_Q_MINUS and index < len(q_minus_texts) else ""
                ),
                "q_plus_text": (
                    q_plus_texts[index] if RAGConfig.ABLATION_Q_PLUS and index < len(q_plus_texts) else ""
                ),
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

        await self._build_hop_edges_from_batch(current_batch)
