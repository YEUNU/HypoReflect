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
    HOPRAG_FORMAT_INSTRUCTION,
    HOPRAG_PROMPT,
    TABLE_TO_TEXT_PROMPT,
)


logger = logging.getLogger(__name__)


class IndexingSupport:
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

    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        text_prompt = HOPRAG_PROMPT.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [{"role": "user", "content": text_prompt}, {"role": "user", "content": HOPRAG_FORMAT_INSTRUCTION}]
        try:
            data = await self.indexing_llm.generate_json(messages, apply_default_sampling=False)
            return {"q_minus": data.get("q_minus", []), "q_plus": data.get("q_plus", []), "summary": data.get("summary", "")}
        except Exception:
            return {"q_minus": [], "q_plus": [], "summary": ""}

    async def _table_to_text(self, table_lines: list[str]) -> list[str]:
        if not table_lines:
            return []

        if not RAGConfig.ABLATION_TABLE_TO_TEXT:
            logger.info("Ablation: Skipping table-to-text conversion.")
            return table_lines

        table_text = "\n".join(table_lines)
        prompt = TABLE_TO_TEXT_PROMPT + f"\nTABLE:\n{table_text}"
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.llm.generate_response(messages, apply_default_sampling=False)
            converted = [sentence.strip() for sentence in response.split("\n") if sentence.strip()]
            if not converted:
                raise ValueError("Empty conversion result")
            return converted
        except Exception as error:
            logger.warning("Table conversion failed (%s), using structured fallback", error)
            fallback: list[str] = []
            headers: list[str] = []
            for line in table_lines:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                if not headers:
                    headers = cells
                else:
                    pairs = [f"{header}: {value}" for header, value in zip(headers, cells) if value and value != "-"]
                    if pairs:
                        fallback.append(", ".join(pairs) + ".")
            return fallback if fallback else table_lines
