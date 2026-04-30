import asyncio
import logging
import re
from typing import Optional

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient, get_llm_client
from utils.prompts import RERANKER_INSTRUCTION
from models.hyporeflect.graphrag_parts import (
    IndexingSupport,
    PipelineSupport,
    QuerySupport,
    RetrievalSupport,
    TextProcessingSupport,
)


logger = logging.getLogger(__name__)


class GraphRAG(TextProcessingSupport, QuerySupport, IndexingSupport, RetrievalSupport, PipelineSupport):
    def __init__(
        self,
        strategy: str = "hyporeflect",
        indexing_model_id: Optional[str] = None,
        corpus_tag: Optional[str] = None,
        save_intermediate: bool = False,
    ):
        self.strategy = strategy.lower()
        self.corpus_tag = corpus_tag or "default"
        self.prefix = self.strategy[:2].upper() + "_"
        self._safe_corpus = re.sub(r"[^A-Za-z0-9_]", "_", self.corpus_tag)

        self.chunk_label = f"{self.prefix}{self._safe_corpus}_Chunk"
        self.doc_label = f"{self.prefix}{self._safe_corpus}_Document"

        self.body_vector_index = f"{self.strategy}_{self._safe_corpus}_vector_idx"
        self.body_text_index = f"{self.strategy}_{self._safe_corpus}_text_idx"
        self.q_minus_vector_index = f"{self.strategy}_{self._safe_corpus}_qminus_vector_idx"
        self.q_plus_vector_index = f"{self.strategy}_{self._safe_corpus}_qplus_vector_idx"
        self.q_minus_text_index = f"{self.strategy}_{self._safe_corpus}_qminus_text_idx"
        self.q_plus_text_index = f"{self.strategy}_{self._safe_corpus}_qplus_text_idx"
        self.vector_index = self.body_vector_index
        self.text_index = self.body_text_index

        self.neo4j = Neo4jService()
        self.llm = VLLMClient()
        self._index_ready = False

        indexing_model_id = indexing_model_id or RAGConfig.DEFAULT_MODEL
        self.indexing_llm = get_llm_client(indexing_model_id)

        self.hop_threshold = RAGConfig.HOP_THRESHOLD
        self.similarity_threshold = RAGConfig.SIMILARITY_THRESHOLD
        self.max_retries = RAGConfig.RETRY_COUNT
        self.vector_dimensions = RAGConfig.EMBEDDING_DIMENSIONS
        self._pending_batch = []
        self._batch_lock = asyncio.Lock()
        self._index_setup_lock = asyncio.Lock()
        self.debug_output_dir = f"data/debug/{self.corpus_tag}"
        self.save_intermediate = save_intermediate

    @staticmethod
    def _reranker_instruction() -> str:
        return RERANKER_INSTRUCTION
