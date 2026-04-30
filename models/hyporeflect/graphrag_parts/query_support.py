import logging
import re
from typing import Any

from core.config import RAGConfig
from utils.prompts import (
    QUERY_REWRITE_FORMAT_INSTRUCTION,
    QUERY_REWRITE_PROMPT,
    SEARCH_CONTINUATION_PROMPT,
)


logger = logging.getLogger(__name__)


class QuerySupport:
    @staticmethod
    def _query_rewrite_prompt() -> str:
        return QUERY_REWRITE_PROMPT

    @staticmethod
    def _search_continuation_prompt() -> str:
        return SEARCH_CONTINUATION_PROMPT

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text.lower()).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(text)
        return unique

    @staticmethod
    def _extract_title_entity_terms(title: str) -> set[str]:
        raw = str(title or "")
        stem = re.split(r"[_\-\s](?:19|20)\d{2}(?:[_\-\s](?:10k|10q|annual|report))?", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        terms = re.findall(r"[A-Za-z][A-Za-z&.\-]{2,}", stem)
        stop = {"inc", "corp", "corporation", "company", "co", "ltd", "plc", "group", "holdings"}
        return {term.lower() for term in terms if term.lower() not in stop}

    def _question_has_entity_token(self, question: str, title: str) -> bool:
        q = str(question or "")
        q_lower = q.lower()
        title_terms = self._extract_title_entity_terms(title)
        if any(term in q_lower for term in title_terms):
            return True

        if re.search(r"\b[A-Z]{2,6}\b", q):
            return True
        return False

    @staticmethod
    def _question_has_period_token(question: str) -> bool:
        q = str(question or "")
        return bool(
            re.search(
                r"\b(?:fy\s?\d{2,4}|fiscal\s+\d{2,4}|(?:19|20)\d{2}|q[1-4]\s?(?:19|20)?\d{2,4}|quarter)\b",
                q,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _question_has_metric_token(question: str) -> bool:
        q_lower = str(question or "").lower()
        metric_terms = [
            "revenue", "net income", "operating income", "operating cash flow",
            "free cash flow", "capex", "capital expenditure", "dividend",
            "eps", "earnings per share", "gross margin", "operating margin",
            "assets", "liabilities", "equity", "property and equipment",
            "depreciation", "amortization", "inventory", "accounts receivable",
            "cash and cash equivalents", "debt", "interest expense",
            "share repurchase", "buyback", "turnover", "ratio",
        ]
        return any(term in q_lower for term in metric_terms)

    @staticmethod
    def _question_has_source_anchor(question: str, chunk_text: str) -> bool:
        markers = [
            "balance sheet", "statement of operations", "income statement",
            "statement of cash flows", "cash flow statement", "footnote", "note ",
            "table", "schedule", "mda", "management discussion",
            "segment", "exhibit", "page ",
        ]
        q_lower = str(question or "").lower()
        chunk_lower = str(chunk_text or "").lower()
        return any(marker in q_lower or marker in chunk_lower for marker in markers)

    @staticmethod
    def _title_surface_forms(title: str) -> set[str]:
        raw = str(title or "").strip()
        if not raw:
            return set()
        forms = {
            raw,
            raw.replace("_", " "),
            re.sub(r"\s+", " ", re.sub(r"[()]", " ", raw.replace("_", " "))).strip(),
        }
        normalized = {
            re.sub(r"\s+", " ", value).strip().lower()
            for value in forms
            if str(value or "").strip()
        }
        return {value for value in normalized if value}

    def _question_mentions_title_surface(self, question: str, title: str) -> bool:
        q_normalized = self._normalize_entity_term(question)
        if not q_normalized:
            return False
        for surface in self._title_surface_forms(title):
            normalized_surface = self._normalize_entity_term(surface)
            if normalized_surface and normalized_surface in q_normalized:
                return True
        return False

    def _is_high_quality_q_plus(self, question: str, title: str, chunk_text: str) -> bool:
        return (
            self._question_has_entity_token(question, title)
            and self._question_has_period_token(question)
            and self._question_has_metric_token(question)
            and self._question_has_source_anchor(question, chunk_text)
        )

    async def _embed_sparse_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        positions: list[int] = []
        payload: list[str] = []
        for index, text in enumerate(texts):
            normalized = str(text or "").strip()
            if not normalized:
                continue
            positions.append(index)
            payload.append(normalized)

        result: list[list[float]] = [[] for _ in texts]
        if not payload:
            return result

        embeddings = await self.llm.get_embeddings(payload)
        for out_index, src_index in enumerate(positions):
            if out_index < len(embeddings):
                result[src_index] = embeddings[out_index]
        return result

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

    @staticmethod
    def _build_context_from_nodes(nodes: list[dict[str, Any]]) -> str:
        return "\n\n".join([
            f"[[{node['title']}, Page {node.get('page', 0)}, Chunk {node['sent_id']}]]\n{node['text']}"
            for node in nodes
        ])

    @staticmethod
    def _node_identity(node: dict[str, Any]) -> str:
        node_id = str(node.get("id", "") or "").strip()
        if node_id:
            return node_id
        return (
            f"{node.get('title', '')}:"
            f"{node.get('source', '')}:"
            f"{node.get('page', 0)}:"
            f"{node.get('sent_id', -1)}"
        )
