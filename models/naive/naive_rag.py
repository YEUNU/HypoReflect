import asyncio
import hashlib
import logging
import os
import re
from typing import List, Dict, Any, Optional

from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient
from core.config import RAGConfig

class NaiveRAG:
    """
    [Baseline] Standard RAG implementation for comparison.
    - Sentence-level Chunking.
    - Standard Vector Search.
    """
    def __init__(self, strategy: str = "naive", corpus_tag: Optional[str] = "default"):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy.lower()
        self.prefix = self.strategy[:2].upper() + "_"
        self.corpus_tag = corpus_tag or "default"
        self.ablation_profile = self._build_ablation_profile()
        branch_token = self._safe_token(f"{self.corpus_tag}_{self.ablation_profile}")
        self.chunk_label = f"{self.prefix}{branch_token}_Chunk"
        self.vector_index = f"{self.strategy}_{branch_token}_vector_idx"
        self.branch_namespace = f"{self.corpus_tag}|{self.ablation_profile}"
        
        self.neo4j = Neo4jService()
        self.vllm = VLLMClient()
        self._index_ready = False
        self._lock = asyncio.Lock()
        self._agentic_runner = None
        self._agentic_full_service = None

    @staticmethod
    def _safe_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
        token = re.sub(r"_+", "_", token).strip("_")
        return token or "default"

    @staticmethod
    def _build_ablation_profile() -> str:
        return (
            f"T{int(RAGConfig.ABLATION_TABLE_TO_TEXT)}"
            f"C{int(RAGConfig.ABLATION_ADAPTIVE_CHUNKING)}"
            f"S{int(RAGConfig.ABLATION_ROLLING_SUMMARY)}"
        )

    async def setup_index(self):
        try:
            await self.neo4j.execute_query(f"""
                CREATE VECTOR INDEX {self.vector_index} IF NOT EXISTS
                FOR (n:{self.chunk_label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }}}}
            """, {"dimensions": RAGConfig.EMBEDDING_DIMENSIONS})
            # Property index for MERGE/MATCH performance
            await self.neo4j.execute_query(
                f"CREATE INDEX {self.chunk_label}_id_idx IF NOT EXISTS FOR (n:{self.chunk_label}) ON (n.id)")
        except Exception as e:
            # EquivalentSchemaRuleAlreadyExists: race condition when multiple workers
            # concurrently hit IF NOT EXISTS — index already exists, safe to ignore.
            if "EquivalentSchemaRuleAlreadyExists" in str(e) or "equivalent index already exists" in str(e).lower():
                self.logger.debug(f"Index already exists (race condition, ignored): {self.vector_index}")
            else:
                self.logger.error(f"Naive Index creation error: {e}")

    def _parse_document(self, content: str) -> tuple:
        lines = content.split("\n")
        title = "Unknown"
        start_idx = 0

        if lines and lines[0].startswith("Title: "):
            title = lines[0].replace("Title: ", "").strip()
            start_idx = 1
        elif lines and lines[0].startswith("Document: "):
            title = lines[0].replace("Document: ", "").strip()
            # Skip OCR header (Document, Pages, separator)
            for i, line in enumerate(lines):
                if "--- Page 1 ---" in line or "=====" in line:
                    start_idx = i + 1
                    break
            if start_idx == 0:
                start_idx = 1

        sentences = lines[start_idx:]
        return title, [s for s in sentences if s.strip()]

    async def index_document(self, filename: str, content: str):
        if not self._index_ready:
            await self.setup_index()
            self._index_ready = True

        title, sentences = self._parse_document(content)
        if not sentences: return
        
        embeddings = await self.vllm.get_embeddings(sentences)
        
        batch_data = []
        for i, (chk, emb) in enumerate(zip(sentences, embeddings)):
            if not emb: continue
            # Namespace by corpus + ablation profile to avoid cross-branch collisions in Neo4j.
            chunk_id = hashlib.md5(
                f"naive|{self.branch_namespace}|{filename}|{title}|{i}".encode()
            ).hexdigest()
            batch_data.append({
                "id": chunk_id,
                "text": chk,
                "source": filename,
                "title": title,
                "sent_id": i,
                "embedding": emb,
                "corpus": self.corpus_tag,
                "branch": self.ablation_profile,
            })

        async with self._lock:
            async with self.neo4j.driver.session() as session:
                query = f"""
                    UNWIND $batch AS item
                    MERGE (c:{self.chunk_label} {{id: item.id, corpus: item.corpus}})
                    SET c.text = item.text,
                        c.source = item.source,
                        c.title = item.title,
                        c.sent_id = item.sent_id,
                        c.corpus = item.corpus,
                        c.branch = item.branch,
                        c.embedding = item.embedding
                """
                await session.run(query, batch=batch_data)  # type: ignore
        
        self.logger.info(f"NaiveRAG: Indexed {len(batch_data)} sentences for {title}")


    async def retrieve(self, query: str, top_k: int = 5) -> tuple:
        query_embedding = await self.vllm.get_embedding(query)
        if not query_embedding:
            return "", []

        async with self.neo4j.driver.session() as session:
            cypher_query = f"""
                CALL db.index.vector.queryNodes('{self.vector_index}', $k, $embedding)
                YIELD node, score
                RETURN node.text as text, node.title as title, node.sent_id as sent_id, node.page as page, score
            """
            result = await session.run(cypher_query, {  # type: ignore
                "k": top_k,
                "embedding": query_embedding,
            })
            
            nodes = [dict(rec) async for rec in result]
        
        context_parts = [f"[[{n['title']}, {n['sent_id']}]]\n{n['text']}" for n in nodes]
        return "\n\n---\n\n".join(context_parts), nodes

    async def run_workflow(self, query: str, history: Optional[List[Dict]] = None) -> tuple:
        """Entry point for benchmark. Returns (answer, sources)."""
        agentic_mode = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
        if agentic_mode == "on":
            agentic_pipeline = str(os.environ.get("RAG_AGENTIC_PIPELINE", "full") or "full").strip().lower()
            if agentic_pipeline == "lite":
                if self._agentic_runner is None:
                    from models.agentic_core import AgenticOrchestrator

                    self._agentic_runner = AgenticOrchestrator(
                        llm=self.vllm,
                        backend=self,
                        strategy_name=self.strategy,
                        top_k=RAGConfig.DEFAULT_TOP_K,
                    )
                return await self._agentic_runner.run(query, history)

            if self._agentic_full_service is None:
                from models.agentic_core import RetrievalGraphAdapter
                from models.hyporeflect.service import AgentService

                graph_adapter = RetrievalGraphAdapter(
                    backend=self,
                    strategy_name=self.strategy,
                    default_top_k=RAGConfig.DEFAULT_TOP_K,
                )
                self._agentic_full_service = AgentService(
                    strategy="hyporeflect",
                    corpus_tag=self.corpus_tag,
                    llm_override=self.vllm,
                    grag_override=graph_adapter,
                )
            return await self._agentic_full_service.run_workflow(query, history)

        _ = history
        context, nodes = await self.retrieve(query)
        
        prompt = f"Answer the following question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        
        answer = await self.vllm.generate_response(messages)
        # Format sources for metric evaluation: {"doc": title, "page": page, "text": text, "sent_id": sent_id}
        trace = [{
            "step": "naive_qa",
            "input": messages,
            "output": answer
        }]
        sources = [
            {
                "doc": n['title'],
                "page": n.get('page', 0),
                "text": n['text'],
                "sent_id": n['sent_id']
            } for n in nodes
        ]
        return answer, sources, trace
