"""Offline indexing pipeline (paper §3.1).

Layer order:
- §3.1.1 Topology-Preserving OCR — runs separately via cli/ocr.py + utils/ocr_tools.
- §3.1.2 Adaptive Context-Aware Chunking — chunking.py
- §3.1.3 Predictive Knowledge Mapping (Q-/Q+) — knowledge_mapping.py
- §3.1.4 Rank-Based HOP Edge Pre-Construction — hop_edges.py
- Neo4j storage (chunks + NEXT edges + index lifecycle) — graph_writer.py
"""
from .chunking import ChunkingMixin
from .graph_writer import GraphWriterMixin
from .hop_edges import HopEdgeMixin
from .knowledge_mapping import KnowledgeMappingMixin


class IndexingPipeline(
    ChunkingMixin,
    KnowledgeMappingMixin,
    HopEdgeMixin,
    GraphWriterMixin,
):
    """Composite mixin exposing the full offline indexing pipeline."""


__all__ = [
    "ChunkingMixin",
    "KnowledgeMappingMixin",
    "HopEdgeMixin",
    "GraphWriterMixin",
    "IndexingPipeline",
]
