import math
import re
from typing import Any, Optional

from models.hyporeflect.stages.common import (
    CITATION_RE,
    NUMERIC_METRIC_KEYS,
    NUMERIC_QUERY_MARKERS,
)


class ContextSupport:
    def _focus_terms(self, query_state: Optional[dict[str, Any]]) -> list[str]:
        if not isinstance(query_state, dict):
            return []

        terms: list[str] = []
        metric = str(query_state.get("metric", "") or "").strip()
        terms.extend(self._metric_alias_terms(metric))

        source_anchor = str(query_state.get("source_anchor", "") or "").strip().lower()
        terms.extend(self._source_anchor_keywords(source_anchor))

        period = str(query_state.get("period", "") or "").strip()
        terms.extend(self._extract_year_tokens(period))

        entity = str(query_state.get("entity", "") or "").strip()
        if entity:
            terms.append(entity.lower())

        deduped: list[str] = []
        seen = set()
        for term in terms:
            key = term.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _extract_relevant_span(
        self,
        text: str,
        query_state: Optional[dict[str, Any]],
        max_chars: int,
    ) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if not clean:
            return ""
        if max_chars <= 0 or len(clean) <= max_chars:
            return clean

        if isinstance(query_state, dict) and str(query_state.get("answer_type", "")).lower() == "compute":
            return clean[:max_chars]

        focus_terms = self._focus_terms(query_state)
        clean_lower = clean.lower()
        hit_idx: Optional[int] = None
        for term in focus_terms:
            idx = clean_lower.find(term)
            if idx >= 0:
                hit_idx = idx
                break

        if hit_idx is None:
            return clean[:max_chars]

        left_margin = max_chars // 3
        start = max(0, hit_idx - left_margin)
        end = min(len(clean), start + max_chars)
        if end - start < max_chars and start > 0:
            start = max(0, end - max_chars)
        return clean[start:end]

    def _value_near_metric_term(
        self,
        *,
        value: str,
        citation_span: str,
        metric_terms: list[str],
    ) -> bool:
        span = str(citation_span or "")
        if not span:
            return False
        lower_span = span.lower()
        value_text = str(value or "").strip()
        if not value_text:
            return False

        normalized_terms: list[str] = []
        for term in metric_terms:
            key = str(term or "").strip().lower()
            if len(key) < 4:
                continue
            normalized_terms.append(key)
        if not normalized_terms:
            return False

        generic_terms = {
            "income",
            "capital",
            "assets",
            "liabilities",
            "shareholders",
            "shareowners",
            "total",
            "metric",
        }
        phrase_terms = [term for term in normalized_terms if " " in term]
        non_generic_terms = [term for term in normalized_terms if term not in generic_terms]
        if phrase_terms:
            scoped_terms = phrase_terms + [term for term in non_generic_terms if term not in phrase_terms]
        elif non_generic_terms:
            scoped_terms = non_generic_terms
        else:
            scoped_terms = normalized_terms

        for term in scoped_terms:
            for match in re.finditer(re.escape(term), lower_span):
                start = max(0, match.start() - 120)
                end = min(len(span), match.end() + 120)
                window = span[start:end]
                if self._value_grounded_in_span(value_text, window):
                    return True
        return False

    def _atom_priority_score(self, atom: dict[str, Any], query_state: dict[str, Any]) -> float:
        text = f"{atom.get('citation', '')} {atom.get('span', '')}".lower()
        if not text.strip():
            return 0.0

        score = 0.0
        metric_terms: list[str] = self._metric_alias_terms(str(query_state.get("metric", "") or ""))
        slot_structs = [
            struct
            for struct in (self._parse_slot_struct(slot) for slot in self._required_slots(query_state))
            if struct
        ]
        for struct in slot_structs:
            metric_terms.extend(self._metric_alias_terms(struct.get("metric", "")))
        metric_terms = list(dict.fromkeys([term.lower() for term in metric_terms if term.strip()]))
        if metric_terms and any(term in text for term in metric_terms):
            score += 2.0

        source_anchors = []
        source_anchor = str(query_state.get("source_anchor", "") or "").strip().lower()
        if source_anchor:
            source_anchors.append(source_anchor)
        source_anchors.extend(
            str(struct.get("source_anchor", "") or "").strip().lower()
            for struct in slot_structs
            if str(struct.get("source_anchor", "") or "").strip()
        )
        anchor_terms: list[str] = []
        for anchor in source_anchors:
            anchor_terms.extend(self._source_anchor_keywords(anchor))
        anchor_terms = list(dict.fromkeys([term.lower() for term in anchor_terms if term.strip()]))
        if anchor_terms and any(term in text for term in anchor_terms):
            score += 3.0

        period_text = str(query_state.get("period", "") or "")
        years = self._extract_year_tokens(period_text)
        if not years:
            for struct in slot_structs:
                years.extend(self._extract_year_tokens(str(struct.get("period", "") or "")))
            years = list(dict.fromkeys(years))
        if years and any(year in text for year in years):
            score += 2.0

        entity = str(query_state.get("entity", "") or "").strip().lower()
        if entity and entity in text:
            score += 1.0

        if str(query_state.get("answer_type", "")).lower() == "extract":
            if any(marker in text for marker in ["expects", "expected", "approximately", "guidance"]):
                score -= 0.5

        return score

    def _slot_atom_alignment_score(self, slot_struct: dict[str, str], atom: dict[str, Any]) -> float:
        text = f"{atom.get('citation', '')} {atom.get('span', '')}".lower()
        if not text.strip():
            return 0.0

        score = 0.0
        metric_terms = [
            term.lower()
            for term in self._metric_alias_terms(slot_struct.get("metric", ""))
            if len(term.strip()) >= 3
        ]
        metric_hits = sum(1 for term in metric_terms if term in text)
        if metric_hits == 0:
            return 0.0
        score += min(3.0, 1.5 + (0.5 * metric_hits))

        period_years = self._extract_year_tokens(str(slot_struct.get("period", "") or ""))
        if period_years:
            year_hits = sum(1 for year in period_years if year in text)
            if year_hits == 0:
                return 0.0
            score += min(2.0, float(year_hits))

        anchor_terms = [
            term.lower()
            for term in self._source_anchor_keywords(slot_struct.get("source_anchor"))
            if len(term.strip()) >= 3
        ]
        if anchor_terms and any(term in text for term in anchor_terms):
            score += 1.0

        slot_entity = str(slot_struct.get("entity", "") or "").strip()
        if slot_entity:
            citation_title = self._citation_doc_title(str(atom.get("citation", "") or ""))
            doc_prefix = citation_title.split("_", 1)[0] if citation_title else ""
            if doc_prefix and self._entity_matches(slot_entity, doc_prefix):
                score += 1.0
        return score

    def _build_context_excerpt(
        self,
        nodes: list[dict[str, Any]],
        limit: int = 8,
        query_state: Optional[dict[str, Any]] = None,
    ) -> str:
        is_compute = (
            isinstance(query_state, dict)
            and str(query_state.get("answer_type", "")).lower() == "compute"
        )
        answer_type = str((query_state or {}).get("answer_type", "")).lower() if isinstance(query_state, dict) else ""
        span_limit = 1400 if is_compute or answer_type == "extract" else 760
        snippets: list[str] = []
        for node in nodes[:limit]:
            title = str(node.get("title") or node.get("doc") or "Unknown")
            page = node.get("page", 0)
            chunk_id = node.get("sent_id", -1)
            text = self._extract_relevant_span(
                str(node.get("text", "") or ""),
                query_state=query_state,
                max_chars=span_limit,
            )
            snippets.append(f"[[{title}, Page {page}, Chunk {chunk_id}]]\n{text}")
        return "\n\n".join(snippets)

    @staticmethod
    def _normalize_nullable_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a"}:
            return None
        return text

    def _normalize_query_entity(self, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            return ""
        if self._is_generic_entity_label(text):
            return ""
        return text

    @staticmethod
    def _normalize_citation(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _coerce_citation(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        direct = CITATION_RE.search(text)
        if direct is not None:
            return direct.group(0).strip()
        loose = re.search(
            r"([A-Za-z0-9_.\-]+)\s*,?\s*Page\s*(\d+)\s*,?\s*Chunk\s*(-?\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if loose is not None:
            title = loose.group(1).strip()
            page = loose.group(2).strip()
            chunk = loose.group(3).strip()
            return f"[[{title}, Page {page}, Chunk {chunk}]]"
        return text

    def _context_citation_map(self, context_excerpt: str) -> dict[str, str]:
        citation_map: dict[str, str] = {}
        if not context_excerpt:
            return citation_map
        pattern = re.compile(
            r"(\[\[[^\]]+,\s*Page\s*\d+\s*,\s*Chunk\s*\d+\s*\]\])\s*\n(.*?)(?=\n\n\[\[|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(context_excerpt):
            citation = self._normalize_citation(match.group(1))
            span = re.sub(r"\s+", " ", match.group(2).strip())
            if citation and span:
                citation_map[citation] = span
        return citation_map

    def _context_excerpt_nodes(self, context_excerpt: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        if not context_excerpt:
            return nodes
        pattern = re.compile(
            r"(\[\[[^\]]+,\s*Page\s*\d+\s*,\s*Chunk\s*-?\d+\s*\]\])\s*\n(.*?)(?=\n\n\[\[|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(context_excerpt):
            citation = self._coerce_citation(match.group(1))
            span = re.sub(r"\s+", " ", str(match.group(2) or "").strip())
            if not citation or not span:
                continue
            citation_meta = re.search(
                r"^\[\[([^,\]]+),\s*Page\s*(\d+)\s*,\s*Chunk\s*(-?\d+)\s*\]\]$",
                citation,
                flags=re.IGNORECASE,
            )
            if citation_meta is None:
                continue
            title = str(citation_meta.group(1) or "").strip()
            try:
                page = int(citation_meta.group(2))
            except Exception:
                page = 0
            try:
                sent_id = int(citation_meta.group(3))
            except Exception:
                sent_id = -1
            nodes.append(
                {
                    "title": title,
                    "page": page,
                    "sent_id": sent_id,
                    "text": span,
                }
            )
        return nodes

    @staticmethod
    def _extract_scaled_numeric_candidates(text: str) -> list[float]:
        raw_text = str(text or "")
        if not raw_text.strip():
            return []

        suffix_factor = {
            "billion": 1e9,
            "bn": 1e9,
            "b": 1e9,
            "million": 1e6,
            "mn": 1e6,
            "mm": 1e6,
            "m": 1e6,
            "thousand": 1e3,
            "k": 1e3,
        }
        candidates: list[float] = []

        def add_candidate(value: float) -> None:
            if not math.isfinite(value):
                return
            rounded = round(float(value), 6)
            candidates.append(rounded)
            candidates.append(round(abs(float(value)), 6))

        pattern = re.compile(
            r"(?P<neg_open>\()?\s*(?P<currency>[$€£])?\s*"
            r"(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
            r"(?P<suffix>billion|million|thousand|bn|mn|mm|b|m|k)?\s*"
            r"(?P<neg_close>\))?",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(raw_text):
            num_raw = str(match.group("num") or "")
            suffix_raw = str(match.group("suffix") or "").strip().lower()
            currency = str(match.group("currency") or "").strip()
            try:
                value = float(num_raw.replace(",", ""))
            except Exception:
                continue

            if match.group("neg_open") and match.group("neg_close"):
                value = -abs(value)

            if suffix_raw in {"m", "b", "k"} and not currency:
                if "." not in num_raw and "," not in num_raw and abs(value) < 100:
                    suffix_raw = ""

            factor = suffix_factor.get(suffix_raw, 1.0)
            add_candidate(value * factor)

        if not candidates:
            for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", raw_text):
                try:
                    add_candidate(float(str(token).replace(",", "")))
                except Exception:
                    continue

        deduped: list[float] = []
        seen = set()
        for num in candidates:
            key = f"{num:.6f}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(num)
        return deduped

    @staticmethod
    def _value_grounded_in_span(value: str, span: str) -> bool:
        value_norm = re.sub(r"\s+", " ", str(value or "").strip().lower())
        span_norm = re.sub(r"\s+", " ", str(span or "").strip().lower())
        if not value_norm or not span_norm:
            return False
        if value_norm in span_norm or span_norm in value_norm:
            return True

        value_nums = ContextSupport._extract_scaled_numeric_candidates(value_norm)
        span_nums = ContextSupport._extract_scaled_numeric_candidates(span_norm)
        if value_nums:
            if not span_nums:
                return False
            for v in value_nums:
                for s in span_nums:
                    tolerance = max(1e-6, abs(s) * 1e-4)
                    if abs(v - s) <= tolerance:
                        return True
            return False

        value_tokens = [tok for tok in re.findall(r"[a-z0-9]{4,}", value_norm)]
        if not value_tokens:
            return False
        span_tokens = set(re.findall(r"[a-z0-9]{4,}", span_norm))
        if not span_tokens:
            return False
        overlap = sum(1 for tok in value_tokens if tok in span_tokens)
        threshold = max(1, int(len(value_tokens) * 0.35))
        return overlap >= threshold

    @staticmethod
    def _contains_numeric_token(text: str) -> bool:
        return re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text or "")) is not None

    @staticmethod
    def _source_anchor_keywords(anchor: Optional[str]) -> list[str]:
        normalized = str(anchor or "").strip().lower()
        if not normalized:
            return []
        terms = [normalized]
        for tok in re.split(r"[^a-z0-9]+", normalized):
            tok = tok.strip()
            if len(tok) >= 3:
                terms.append(tok)
        deduped: list[str] = []
        seen = set()
        for term in terms:
            key = term.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    @staticmethod
    def _source_anchor_strict_markers(anchor: Optional[str]) -> list[str]:
        normalized = str(anchor or "").strip().lower()
        if not normalized:
            return []
        if normalized == "cash flow statement":
            return [
                "cash flow statement",
                "statement of cash flows",
                "consolidated statement of cash flows",
                "cash flows from operating activities",
                "cash flows from investing activities",
                "cash flows from financing activities",
                "net cash provided by operating activities",
                "net cash provided by investing activities",
                "net cash used in financing activities",
                "net increase (decrease) in cash and cash equivalents",
                "purchases of property, plant and equipment",
                "purchases of pp&e",
            ]
        if normalized == "income statement":
            return [
                "income statement",
                "statement of income",
                "statement of operations",
                "consolidated statements of operations",
                "consolidated statement of income",
            ]
        if normalized == "balance sheet":
            return [
                "balance sheet",
                "statement of financial position",
                "consolidated balance sheets",
                "consolidated balance sheet",
            ]
        return []

    def _is_anchor_grounded_in_citation(
        self,
        source_anchor: Optional[str],
        citation: str,
        citation_span: str,
    ) -> bool:
        markers = self._source_anchor_strict_markers(source_anchor)
        if not markers:
            return True
        blob = f"{citation} {citation_span}".lower()
        return any(marker in blob for marker in markers)

    @staticmethod
    def _extract_boolean_label(answer: str) -> Optional[str]:
        text = str(answer or "")
        text = re.sub(r"\[\[[^\]]+\]\]", "", text)
        text = re.sub(r"(?i)@@answer:\s*", "", text, count=1).strip().lower()
        if re.match(r"^(yes|no)\b", text):
            return text.split()[0]
        if re.match(r"^(true|false)\b", text):
            return "yes" if text.startswith("true") else "no"
        return None

    @staticmethod
    def _extract_year_tokens(text: str) -> list[str]:
        raw = re.findall(
            r"(?<!\d)(?:fy\s*)?((?:19|20)\d{2})(?!\d)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        years: list[str] = []
        seen = set()
        for year in raw:
            if year not in seen:
                seen.add(year)
                years.append(year)
        return years

    @staticmethod
    def _period_granularity(period: Any) -> str:
        text = str(period or "").strip().lower()
        if not text:
            return "unknown"
        if re.search(r"\bq[1-4]\b", text) or "quarter" in text or "three months" in text:
            return "quarter"
        if "fy" in text or "fiscal year" in text or "year end" in text or re.search(r"\b(?:19|20)\d{2}\b", text):
            return "annual"
        return "unknown"

    @staticmethod
    def _has_quarter_markers(text: str) -> bool:
        blob = str(text or "").lower()
        if not blob:
            return False
        markers = [
            "three months ended",
            "quarter ended",
            "quarterly",
            "q1",
            "q2",
            "q3",
            "q4",
            "first quarter",
            "second quarter",
            "third quarter",
            "fourth quarter",
        ]
        return any(marker in blob for marker in markers)

    @staticmethod
    def _has_annual_markers(text: str) -> bool:
        blob = str(text or "").lower()
        if not blob:
            return False
        markers = [
            "year ended",
            "twelve months ended",
            "fiscal year",
            "annual",
        ]
        return any(marker in blob for marker in markers)

    def _value_matches_slot_period(
        self,
        *,
        value: str,
        slot_period: str,
        citation: str,
        citation_span: str,
    ) -> bool:
        period_text = str(slot_period or "").strip()
        span_text = str(citation_span or "")
        if not period_text or not span_text:
            return True

        year_aligned = True
        slot_years = set(self._extract_year_tokens(period_text))
        span_lower = span_text.lower()
        if slot_years:
            if any(year in span_lower for year in slot_years):
                year_aligned = False
                for year in slot_years:
                    for year_match in re.finditer(re.escape(year), span_lower):
                        start = max(0, year_match.start() - 180)
                        end = min(len(span_text), year_match.end() + 180)
                        window = span_text[start:end]
                        if self._value_grounded_in_span(value, window):
                            year_aligned = True
                            break
                    if year_aligned:
                        break
                if not year_aligned:
                    return False
            title_years = set(self._extract_year_tokens(self._citation_doc_title(citation)))
            if title_years and title_years.isdisjoint(slot_years):
                return False

        granularity = self._period_granularity(period_text)
        if granularity == "annual":
            has_quarter = self._has_quarter_markers(span_text)
            has_annual = self._has_annual_markers(span_text)
            if has_quarter and not has_annual:
                return False
        if granularity == "quarter":
            has_quarter = self._has_quarter_markers(span_text)
            has_annual = self._has_annual_markers(span_text)
            if has_annual and not has_quarter:
                return False
        return year_aligned

    @staticmethod
    def _extract_query_statement_anchors(query: str) -> list[str]:
        query_lower = str(query or "").strip().lower()
        if not query_lower:
            return []

        anchors: list[str] = []

        def add(anchor: str) -> None:
            if anchor not in anchors:
                anchors.append(anchor)

        balance_markers = [
            "balance sheet",
            "balance sheets",
            "statement of financial position",
        ]
        income_markers = [
            "income statement",
            "income statements",
            "statement of income",
            "statement of operations",
            "p&l",
            "p & l",
        ]
        cashflow_markers = [
            "cash flow statement",
            "cash flow statements",
            "statement of cash flows",
        ]

        if any(marker in query_lower for marker in balance_markers):
            add("balance sheet")
        if any(marker in query_lower for marker in income_markers):
            add("income statement")
        if any(marker in query_lower for marker in cashflow_markers):
            add("cash flow statement")
        return anchors

    @staticmethod
    def _query_has_explicit_statement_anchor(query: str) -> bool:
        return bool(ContextSupport._extract_query_statement_anchors(query))

    def _infer_anchor_for_metric(self, metric: Any) -> Optional[str]:
        metric_key = self._canonical_metric_key(metric)
        if not metric_key:
            return None

        if any(
            token in metric_key
            for token in [
                "capital expenditure",
                "capex",
                "depreciation",
                "amortization",
                "dividend",
                "cash from operations",
                "operating cash flow",
                "cash flows from",
                "financing activities",
                "investing activities",
            ]
        ):
            return "cash flow statement"
        if any(
            token in metric_key
            for token in [
                "revenue",
                "net sales",
                "gross profit",
                "operating income",
                "operating margin",
                "net income",
                "ebit",
                "ebitda",
                "cogs",
                "cost of goods sold",
                "eps",
                "earnings per share",
            ]
        ):
            return "income statement"
        if any(
            token in metric_key
            for token in [
                "total assets",
                "total liabilities",
                "current assets",
                "current liabilities",
                "accounts payable",
                "accounts receivable",
                "inventory",
                "property and equipment",
                "property, plant and equipment",
                "pp&e",
                "net ppne",
                "net ar",
                "quick ratio",
                "liability",
            ]
        ):
            return "balance sheet"
        return None

    def _is_numeric_compute_query(self, query: str, query_state: dict[str, Any]) -> bool:
        if str(query_state.get("answer_type", "")).lower() != "compute":
            return False
        query_lower = str(query or "").lower()
        metric_key = self._canonical_metric_key(query_state.get("metric", ""))
        if any(marker in query_lower for marker in NUMERIC_QUERY_MARKERS):
            return True
        return metric_key in NUMERIC_METRIC_KEYS
