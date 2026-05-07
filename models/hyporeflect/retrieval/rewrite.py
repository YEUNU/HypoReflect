"""Query rewriting for retrieval (paper §3.2.3 "Query Rewriting and Forced Synthesis").

N_r = 2 rewrites with weight w_r = 0.85 (RAGConfig.QUERY_REWRITE_COUNT,
RAGConfig.QUERY_REWRITE_WEIGHT). The rewrite is invoked before graph search
when slot fills are insufficient.
"""
import logging

from core.config import RAGConfig
from utils.prompts import (
    QUERY_REWRITE_FORMAT_INSTRUCTION,
    QUERY_REWRITE_PROMPT,
    SEARCH_CONTINUATION_PROMPT,
)


logger = logging.getLogger(__name__)


class QueryRewriteMixin:
    @staticmethod
    def _query_rewrite_prompt() -> str:
        return QUERY_REWRITE_PROMPT

    @staticmethod
    def _search_continuation_prompt() -> str:
        return SEARCH_CONTINUATION_PROMPT

    async def _rewrite_query(self, query: str) -> list[str]:
        if not query:
            return []
        messages = [
            {"role": "user", "content": self._query_rewrite_prompt().format(query=query)},
            {"role": "user", "content": QUERY_REWRITE_FORMAT_INSTRUCTION},
        ]
        try:
            data = await self.llm.generate_json(messages, apply_default_sampling=False)
            rewrites = data.get("positive_queries", []) if isinstance(data, dict) else []
            if not isinstance(rewrites, list):
                return []
            unique: list[str] = []
            seen: set[str] = set()
            for rewrite in rewrites:
                if not isinstance(rewrite, str):
                    continue
                normalized = self._normalize_entity_term(rewrite)
                if not normalized:
                    continue
                if normalized == self._normalize_entity_term(query):
                    continue
                if normalized in seen:
                    continue
                unique.append(rewrite.strip())
                seen.add(normalized)
            return unique[: max(0, RAGConfig.QUERY_REWRITE_COUNT)]
        except Exception as error:
            logger.warning("Query rewrite failed: %s", error)
            return []
