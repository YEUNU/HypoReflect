"""Text normalization and query metadata helpers.

Used by retrieval to derive entity/period/doc-type signals from a query and
calibrate ranking with meta boosts and boilerplate penalties.
"""
import re
import unicodedata
from typing import Any


class TextUtilsMixin:
    @staticmethod
    def _normalize_entity_term(value: str) -> str:
        """Normalize entity/query tokens for robust graph matching."""
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKC", str(value)).lower()
        normalized = re.sub(r"[_\-]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _sanitize_fulltext_query(value: str) -> str:
        """Sanitize free-form text for Neo4j Lucene fulltext query parser."""
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKC", str(value))
        normalized = re.sub(r"[+\-!(){}\[\]^\"~*?:\\/|&]", " ", normalized)
        normalized = re.sub(r"[`]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return ""
        return normalized[:512]

    @staticmethod
    def _normalize_doc_key(value: str) -> str:
        if not value:
            return ""
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    @staticmethod
    def _clean_company_candidate(candidate: str) -> str:
        if not candidate:
            return ""
        cleaned = str(candidate).lower()
        cleaned = re.split(r"[?.!,:;]", cleaned)[0].strip()
        cleaned = re.sub(
            r"\b(as of|as at|between|during|from|to|in|on|at|for|by)\b.*$",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(r"\b(fy|fiscal|year|years|q[1-4]|quarter)\b.*$", "", cleaned).strip()
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        return cleaned

    @staticmethod
    def _extract_named_entities(query: str) -> list[str]:
        text = str(query or "").strip()
        if not text:
            return []
        pattern = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9'&.\-]*)(?:\s+(?:[A-Z][A-Za-z0-9'&.\-]*|of|the|and|for|in|on|to|&))*"
        )
        stopwords = {
            "what", "which", "who", "when", "where", "why", "how",
            "is", "are", "was", "were", "do", "does", "did",
            "among", "between", "list", "name",
        }
        entities: list[str] = []
        seen = set()

        def add_entity(value: str) -> None:
            key = value.lower()
            if not value or key in seen:
                return
            seen.add(key)
            entities.append(value)

        for match in pattern.finditer(text):
            raw = str(match.group(0) or "").strip(" ,?.!;:()[]{}\"'")
            if not raw:
                continue
            words = [w for w in raw.split() if w]
            while words and words[0].lower() in stopwords:
                words = words[1:]
            while words and words[-1].lower() in {"of", "the", "and", "for", "in", "on", "to", "&"}:
                words = words[:-1]
            if not words:
                continue
            entity = " ".join(words).strip()
            if not entity:
                continue
            if " and " in entity.lower():
                parts = [part.strip() for part in re.split(r"\band\b", entity, flags=re.IGNORECASE) if part.strip()]
                if len(parts) >= 2 and all(len(part.split()) >= 2 for part in parts):
                    for part in parts:
                        add_entity(part)
                    continue
            add_entity(entity)
        return entities

    @staticmethod
    def _entity_term_tokens(value: str) -> set[str]:
        normalized = TextUtilsMixin._normalize_entity_term(value)
        return {
            tok for tok in re.findall(r"[a-z0-9]+", normalized)
            if tok not in {"the", "of", "and", "for", "in", "on", "to", "a", "an"}
        }

    def _node_matches_named_entity(self, node: dict[str, Any], entity: str) -> bool:
        title = str(node.get("title", "") or "")
        text = str(node.get("text", "") or "")
        title_key = self._normalize_doc_key(title)
        entity_key = self._normalize_doc_key(entity)
        if entity_key and (entity_key in title_key or title_key in entity_key):
            return True
        entity_terms = self._entity_term_tokens(entity)
        title_terms = self._entity_term_tokens(title)
        if entity_terms and title_terms and not entity_terms.isdisjoint(title_terms):
            return True
        text_lower = text.lower()
        entity_lower = str(entity or "").strip().lower()
        return bool(entity_lower and entity_lower in text_lower)

    def _extract_company_keys(self, query: str) -> set[str]:
        q = query or ""
        q_lower = q.lower()
        company_candidates: list[str] = []

        for match in re.finditer(r"\b([a-z0-9&.,()\-]{2,40}(?:\s+[a-z0-9&.,()\-]{1,40}){0,3})'s\b", q_lower):
            company_candidates.append(match.group(1).strip())

        for pattern in [
            r"\bfor\s+([a-z0-9&.,'()\- ]{2,80})",
            r"\bof\s+([a-z0-9&.,'()\- ]{2,80})",
        ]:
            match = re.search(pattern, q_lower)
            if match:
                company_candidates.append(match.group(1).strip())

        ticker_stopwords = {
            "FY", "USD", "GAAP", "NON", "Q1", "Q2", "Q3", "Q4",
            "COGS", "ROA", "DPO", "PPNE", "PPE", "AR", "YOY",
        }
        for tok in re.findall(r"\b[A-Z]{2,6}\b", q):
            upper = tok.strip().upper()
            if upper in ticker_stopwords:
                continue
            company_candidates.append(tok.strip())

        company_keys: set[str] = set()
        for raw in company_candidates:
            cleaned = self._clean_company_candidate(raw)
            if len(cleaned) < 2:
                continue
            normalized = self._normalize_doc_key(cleaned)
            if normalized:
                company_keys.add(normalized)

            first_token = cleaned.split()[0] if cleaned.split() else ""
            first_key = self._normalize_doc_key(first_token)
            if first_key and len(first_key) >= 2:
                company_keys.add(first_key)

        return company_keys

    def _node_matches_company(self, node: dict[str, Any], query_meta: dict[str, Any]) -> bool:
        company_keys = set(query_meta.get("company_keys") or [])
        legacy_key = str(query_meta.get("company_key", "") or "").strip()
        if legacy_key:
            company_keys.add(legacy_key)
        if not company_keys:
            return True
        title_key = self._normalize_doc_key(str(node.get("title", "") or ""))
        return any(key and key in title_key for key in company_keys)

    def _extract_query_metadata(self, query: str) -> dict[str, Any]:
        q = query or ""
        q_lower = q.lower()
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", q_lower))

        doc_types = set()
        if re.search(r"\b10[\s\-]?k\b", q_lower):
            doc_types.add("10k")
        if re.search(r"\b10[\s\-]?q\b", q_lower):
            doc_types.add("10q")

        company_keys = self._extract_company_keys(q)
        company_key = sorted(company_keys)[0] if company_keys else ""
        financial_intent = any(
            kw in q_lower for kw in [
                "ratio", "revenue", "balance sheet", "cash flow", "statement of",
                "income", "assets", "liabilities", "pp&e", "capex", "fiscal", "fy",
                "percent", "%", "turnover"
            ]
        )
        return {
            "years": years,
            "doc_types": doc_types,
            "company_key": company_key,
            "company_keys": company_keys,
            "financial_intent": financial_intent,
        }

    def _meta_boost_for_node(self, node: dict[str, Any], query_meta: dict[str, Any]) -> float:
        title = str(node.get("title", "") or "")
        text = str(node.get("text", "") or "")
        title_lower = title.lower()
        boost = 0.0

        for year in query_meta.get("years", set()):
            if year in title_lower:
                boost += 0.25
                break

        for dtype in query_meta.get("doc_types", set()):
            dtype_key = dtype.replace("-", "")
            if dtype_key in title_lower.replace("-", "").replace("_", ""):
                boost += 0.15
                break

        company_keys = set(query_meta.get("company_keys") or [])
        legacy_key = str(query_meta.get("company_key", "") or "").strip()
        has_company_constraints = bool(company_keys or legacy_key)
        if has_company_constraints and self._node_matches_company(node, query_meta):
            boost += 0.35

        if query_meta.get("financial_intent", False):
            text_lower = text.lower()
            finance_markers = [
                "consolidated balance sheets",
                "consolidated statements of operations",
                "consolidated statements of cash flows",
                "property and equipment",
                "net revenues",
                "statement of income",
                "statement of financial position",
            ]
            if any(marker in text_lower for marker in finance_markers):
                boost += 0.15

        return min(boost, 0.9)

    def _company_mismatch_penalty(self, node: dict[str, Any], query_meta: dict[str, Any]) -> float:
        company_keys = set(query_meta.get("company_keys") or [])
        legacy_key = str(query_meta.get("company_key", "") or "").strip()
        if legacy_key:
            company_keys.add(legacy_key)
        if not company_keys:
            return 0.0
        if self._node_matches_company(node, query_meta):
            return 0.0
        return 0.45 if query_meta.get("financial_intent", False) else 0.25

    def _apply_retrieval_calibration(self, nodes: list[dict[str, Any]], query_meta: dict[str, Any]) -> None:
        for node in nodes:
            node["meta_boost"] = self._meta_boost_for_node(node, query_meta)
            base_penalty = self._boilerplate_penalty(str(node.get("text", "") or ""))
            mismatch_penalty = self._company_mismatch_penalty(node, query_meta)
            node["company_mismatch_penalty"] = mismatch_penalty
            node["boilerplate_penalty"] = min(1.0, base_penalty + mismatch_penalty)

    def _boilerplate_penalty(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        patterns = [
            "forward-looking statement",
            "forward looking statement",
            "risk factors",
            "we caution",
            "could cause our actual",
            "subject to risks and uncertainties",
            "pending merger",
            "management's current expectations",
        ]
        penalty = 0.0
        for pattern in patterns:
            if pattern in text_lower:
                penalty += 0.18

        if any(marker in text_lower for marker in [
            "consolidated balance sheets",
            "consolidated statements of operations",
            "consolidated statements of cash flows",
            "property and equipment, net",
            "total net revenues",
        ]):
            penalty -= 0.2

        return max(0.0, min(0.8, penalty))

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
