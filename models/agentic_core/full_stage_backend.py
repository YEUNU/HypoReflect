from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from core.config import RAGConfig


logger = logging.getLogger(__name__)


class RetrievalGraphAdapter:
    """Adapt a model's `retrieve(query, top_k)` API to HypoReflect Execution's `graph_search` API."""

    def __init__(
        self,
        backend: Any,
        strategy_name: str,
        default_top_k: int | None = None,
    ) -> None:
        self.backend = backend
        self.strategy_name = strategy_name
        base_k = default_top_k if default_top_k is not None else RAGConfig.DEFAULT_TOP_K
        self.default_top_k = max(1, int(base_k))

    async def graph_search(self, entities: List[str], depth: int = 2, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        _ = depth
        query_parts = [str(e).strip() for e in (entities or []) if str(e).strip()]
        query = " ".join(query_parts).strip()
        if not query:
            return "", []

        k = max(self.default_top_k, int(top_k or self.default_top_k))
        try:
            context, nodes = await self.backend.retrieve(query, top_k=k)
        except TypeError:
            # Defensive fallback when adapter backend ignores top_k argument.
            context, nodes = await self.backend.retrieve(query)
        except Exception as e:
            logger.warning("[%s] graph adapter retrieve failed: %s", self.strategy_name, e)
            return "", []

        normalized_nodes = self._normalize_nodes(nodes)
        context_text = str(context or "").strip()
        if not context_text:
            context_text = self._context_from_nodes(normalized_nodes)
        return context_text, normalized_nodes

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _normalize_nodes(self, nodes: Any) -> List[Dict[str, Any]]:
        if not isinstance(nodes, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            text = str(node.get("text", "") or "").strip()
            if not text:
                continue
            title = str(
                node.get("title")
                or node.get("doc")
                or node.get("source")
                or "Unknown"
            ).strip() or "Unknown"
            page = self._safe_int(node.get("page", 0), 0)
            sent_id = self._safe_int(node.get("sent_id", node.get("chunk", idx)), idx)
            normalized.append(
                {
                    "title": title,
                    "doc": title,
                    "page": page,
                    "sent_id": sent_id,
                    "text": text,
                }
            )
        return normalized

    @staticmethod
    def _context_from_nodes(nodes: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for node in nodes:
            title = str(node.get("title", "Unknown") or "Unknown")
            page = int(node.get("page", 0) or 0)
            sent_id = int(node.get("sent_id", 0) or 0)
            text = str(node.get("text", "") or "")
            blocks.append(f"[[{title}, Page {page}, Chunk {sent_id}]]\n{text}")
        return "\n\n---\n\n".join(blocks)
