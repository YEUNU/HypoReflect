"""Retrieval layer (paper §3.2.3 Execution).

Three index types per chunk are consulted (Body, Q-, Q+); two stages combine
them via reciprocal rank fusion; reranking gates with tau_r=0.5; pre-built
NEXT/HOP edges drive graph traversal; queries may be rewritten N_r=2 times
with weight w_r=0.85.

Modules:
- text_utils.py — normalization, query metadata, boilerplate penalty
- quality_gates.py — Q+ quality gating + sparse-text embedding helpers
- rewrite.py — query rewrite (N_r, w_r)
- hybrid.py — RRF over {body, q_minus, q_plus} channels
- rerank.py — cross-encoder rerank + tau_r threshold + meta calibration
- traversal.py — graph_search over NEXT/HOP edges (offline / runtime modes)
- retrieve.py — two-stage Q-/Q+ retrieve entry point
"""
from .hybrid import HybridSearchMixin
from .quality_gates import QualityGatesMixin
from .rerank import RerankMixin
from .retrieve import RetrieveMixin
from .rewrite import QueryRewriteMixin
from .text_utils import TextUtilsMixin
from .traversal import TraversalMixin


class RetrievalPipeline(
    TextUtilsMixin,
    QualityGatesMixin,
    QueryRewriteMixin,
    HybridSearchMixin,
    RerankMixin,
    TraversalMixin,
    RetrieveMixin,
):
    """Composite mixin exposing the full query-time retrieval pipeline."""


__all__ = [
    "TextUtilsMixin",
    "QualityGatesMixin",
    "QueryRewriteMixin",
    "HybridSearchMixin",
    "RerankMixin",
    "TraversalMixin",
    "RetrieveMixin",
    "RetrievalPipeline",
]
