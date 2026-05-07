"""Context excerpt building, atomization, packing, numeric/period validation
(paper §3.2.3 evidence-context preparation)."""
from typing import Any
from typing import Any, Optional
import logging
import math
import re
from models.hyporeflect.stages.common import CITATION_RE
from models.hyporeflect.stages.common import CITATION_RE, NUMERIC_METRIC_KEYS, NUMERIC_QUERY_MARKERS
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace
from utils.prompts import AGENT_EXECUTION_SYSTEM_PROMPT, COMPLEX_AGENT_PROMPT_TEMPLATE, CONTEXT_ATOMIZATION_FORMAT_INSTRUCTION, CONTEXT_ATOMIZATION_PROMPT, CONTEXT_ATOMIZATION_RETRY_PROMPT, CONTEXT_PACKING_FORMAT_INSTRUCTION, CONTEXT_PACKING_PROMPT, CONTEXT_PACKING_RETRY_PROMPT, QUERY_STATE_PROMPT, QUERY_STATE_REVIEW_PROMPT
logger = logging.getLogger(__name__)

class ContextSupport:

    def _focus_terms(self, query_state: Optional[dict[str, Any]]) -> list[str]:
        if not isinstance(query_state, dict):
            return []
        terms: list[str] = []
        metric = str(query_state.get('metric', '') or '').strip()
        terms.extend(self._metric_alias_terms(metric))
        source_anchor = str(query_state.get('source_anchor', '') or '').strip().lower()
        terms.extend(self._source_anchor_keywords(source_anchor))
        period = str(query_state.get('period', '') or '').strip()
        terms.extend(self._extract_year_tokens(period))
        entity = str(query_state.get('entity', '') or '').strip()
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

    def _extract_relevant_span(self, text: str, query_state: Optional[dict[str, Any]], max_chars: int) -> str:
        clean = re.sub('\\s+', ' ', str(text or '')).strip()
        if not clean:
            return ''
        if max_chars <= 0 or len(clean) <= max_chars:
            return clean
        if isinstance(query_state, dict) and str(query_state.get('answer_type', '')).lower() == 'compute':
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

    def _value_near_metric_term(self, *, value: str, citation_span: str, metric_terms: list[str]) -> bool:
        span = str(citation_span or '')
        if not span:
            return False
        lower_span = span.lower()
        value_text = str(value or '').strip()
        if not value_text:
            return False
        normalized_terms: list[str] = []
        for term in metric_terms:
            key = str(term or '').strip().lower()
            if len(key) < 4:
                continue
            normalized_terms.append(key)
        if not normalized_terms:
            return False
        generic_terms = {'income', 'capital', 'assets', 'liabilities', 'shareholders', 'shareowners', 'total', 'metric'}
        phrase_terms = [term for term in normalized_terms if ' ' in term]
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
        metric_terms: list[str] = self._metric_alias_terms(str(query_state.get('metric', '') or ''))
        slot_structs = [struct for struct in (self._parse_slot_struct(slot) for slot in self._required_slots(query_state)) if struct]
        for struct in slot_structs:
            metric_terms.extend(self._metric_alias_terms(struct.get('metric', '')))
        metric_terms = list(dict.fromkeys([term.lower() for term in metric_terms if term.strip()]))
        if metric_terms and any((term in text for term in metric_terms)):
            score += 2.0
        source_anchors = []
        source_anchor = str(query_state.get('source_anchor', '') or '').strip().lower()
        if source_anchor:
            source_anchors.append(source_anchor)
        source_anchors.extend((str(struct.get('source_anchor', '') or '').strip().lower() for struct in slot_structs if str(struct.get('source_anchor', '') or '').strip()))
        anchor_terms: list[str] = []
        for anchor in source_anchors:
            anchor_terms.extend(self._source_anchor_keywords(anchor))
        anchor_terms = list(dict.fromkeys([term.lower() for term in anchor_terms if term.strip()]))
        if anchor_terms and any((term in text for term in anchor_terms)):
            score += 3.0
        period_text = str(query_state.get('period', '') or '')
        years = self._extract_year_tokens(period_text)
        if not years:
            for struct in slot_structs:
                years.extend(self._extract_year_tokens(str(struct.get('period', '') or '')))
            years = list(dict.fromkeys(years))
        if years and any((year in text for year in years)):
            score += 2.0
        entity = str(query_state.get('entity', '') or '').strip().lower()
        if entity and entity in text:
            score += 1.0
        if str(query_state.get('answer_type', '')).lower() == 'extract':
            if any((marker in text for marker in ['expects', 'expected', 'approximately', 'guidance'])):
                score -= 0.5
        return score

    def _slot_atom_alignment_score(self, slot_struct: dict[str, str], atom: dict[str, Any]) -> float:
        text = f"{atom.get('citation', '')} {atom.get('span', '')}".lower()
        if not text.strip():
            return 0.0
        score = 0.0
        metric_terms = [term.lower() for term in self._metric_alias_terms(slot_struct.get('metric', '')) if len(term.strip()) >= 3]
        metric_hits = sum((1 for term in metric_terms if term in text))
        if metric_hits == 0:
            return 0.0
        score += min(3.0, 1.5 + 0.5 * metric_hits)
        period_years = self._extract_year_tokens(str(slot_struct.get('period', '') or ''))
        if period_years:
            year_hits = sum((1 for year in period_years if year in text))
            if year_hits == 0:
                return 0.0
            score += min(2.0, float(year_hits))
        anchor_terms = [term.lower() for term in self._source_anchor_keywords(slot_struct.get('source_anchor')) if len(term.strip()) >= 3]
        if anchor_terms and any((term in text for term in anchor_terms)):
            score += 1.0
        slot_entity = str(slot_struct.get('entity', '') or '').strip()
        if slot_entity:
            citation_title = self._citation_doc_title(str(atom.get('citation', '') or ''))
            doc_prefix = citation_title.split('_', 1)[0] if citation_title else ''
            if doc_prefix and self._entity_matches(slot_entity, doc_prefix):
                score += 1.0
        return score

    def _build_context_excerpt(self, nodes: list[dict[str, Any]], limit: int=8, query_state: Optional[dict[str, Any]]=None) -> str:
        is_compute = isinstance(query_state, dict) and str(query_state.get('answer_type', '')).lower() == 'compute'
        answer_type = str((query_state or {}).get('answer_type', '')).lower() if isinstance(query_state, dict) else ''
        span_limit = 1400 if is_compute or answer_type == 'extract' else 760
        snippets: list[str] = []
        for node in nodes[:limit]:
            title = str(node.get('title') or node.get('doc') or 'Unknown')
            page = node.get('page', 0)
            chunk_id = node.get('sent_id', -1)
            text = self._extract_relevant_span(str(node.get('text', '') or ''), query_state=query_state, max_chars=span_limit)
            snippets.append(f'[[{title}, Page {page}, Chunk {chunk_id}]]\n{text}')
        return '\n\n'.join(snippets)

    @staticmethod
    def _normalize_nullable_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {'null', 'none', 'n/a'}:
            return None
        return text

    def _normalize_query_entity(self, value: Any) -> str:
        text = re.sub('\\s+', ' ', str(value or '').strip())
        if not text:
            return ''
        if self._is_generic_entity_label(text):
            return ''
        return text

    @staticmethod
    def _normalize_citation(value: Any) -> str:
        return re.sub('\\s+', ' ', str(value or '').strip().lower())

    def _coerce_citation(self, value: Any) -> str:
        text = str(value or '').strip()
        if not text:
            return ''
        direct = CITATION_RE.search(text)
        if direct is not None:
            return direct.group(0).strip()
        loose = re.search('([A-Za-z0-9_.\\-]+)\\s*,?\\s*Page\\s*(\\d+)\\s*,?\\s*Chunk\\s*(-?\\d+)', text, flags=re.IGNORECASE)
        if loose is not None:
            title = loose.group(1).strip()
            page = loose.group(2).strip()
            chunk = loose.group(3).strip()
            return f'[[{title}, Page {page}, Chunk {chunk}]]'
        return text

    def _context_citation_map(self, context_excerpt: str) -> dict[str, str]:
        citation_map: dict[str, str] = {}
        if not context_excerpt:
            return citation_map
        pattern = re.compile('(\\[\\[[^\\]]+,\\s*Page\\s*\\d+\\s*,\\s*Chunk\\s*\\d+\\s*\\]\\])\\s*\\n(.*?)(?=\\n\\n\\[\\[|\\Z)', flags=re.IGNORECASE | re.DOTALL)
        for match in pattern.finditer(context_excerpt):
            citation = self._normalize_citation(match.group(1))
            span = re.sub('\\s+', ' ', match.group(2).strip())
            if citation and span:
                citation_map[citation] = span
        return citation_map

    def _context_excerpt_nodes(self, context_excerpt: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        if not context_excerpt:
            return nodes
        pattern = re.compile('(\\[\\[[^\\]]+,\\s*Page\\s*\\d+\\s*,\\s*Chunk\\s*-?\\d+\\s*\\]\\])\\s*\\n(.*?)(?=\\n\\n\\[\\[|\\Z)', flags=re.IGNORECASE | re.DOTALL)
        for match in pattern.finditer(context_excerpt):
            citation = self._coerce_citation(match.group(1))
            span = re.sub('\\s+', ' ', str(match.group(2) or '').strip())
            if not citation or not span:
                continue
            citation_meta = re.search('^\\[\\[([^,\\]]+),\\s*Page\\s*(\\d+)\\s*,\\s*Chunk\\s*(-?\\d+)\\s*\\]\\]$', citation, flags=re.IGNORECASE)
            if citation_meta is None:
                continue
            title = str(citation_meta.group(1) or '').strip()
            try:
                page = int(citation_meta.group(2))
            except Exception:
                page = 0
            try:
                sent_id = int(citation_meta.group(3))
            except Exception:
                sent_id = -1
            nodes.append({'title': title, 'page': page, 'sent_id': sent_id, 'text': span})
        return nodes

    @staticmethod
    def _extract_scaled_numeric_candidates(text: str) -> list[float]:
        raw_text = str(text or '')
        if not raw_text.strip():
            return []
        suffix_factor = {'billion': 1000000000.0, 'bn': 1000000000.0, 'b': 1000000000.0, 'million': 1000000.0, 'mn': 1000000.0, 'mm': 1000000.0, 'm': 1000000.0, 'thousand': 1000.0, 'k': 1000.0}
        candidates: list[float] = []

        def add_candidate(value: float) -> None:
            if not math.isfinite(value):
                return
            rounded = round(float(value), 6)
            candidates.append(rounded)
            candidates.append(round(abs(float(value)), 6))
        pattern = re.compile('(?P<neg_open>\\()?\\s*(?P<currency>[$€£])?\\s*(?P<num>[-+]?\\d[\\d,]*(?:\\.\\d+)?)\\s*(?P<suffix>billion|million|thousand|bn|mn|mm|b|m|k)?\\s*(?P<neg_close>\\))?', flags=re.IGNORECASE)
        for match in pattern.finditer(raw_text):
            num_raw = str(match.group('num') or '')
            suffix_raw = str(match.group('suffix') or '').strip().lower()
            currency = str(match.group('currency') or '').strip()
            try:
                value = float(num_raw.replace(',', ''))
            except Exception:
                continue
            if match.group('neg_open') and match.group('neg_close'):
                value = -abs(value)
            if suffix_raw in {'m', 'b', 'k'} and (not currency):
                if '.' not in num_raw and ',' not in num_raw and (abs(value) < 100):
                    suffix_raw = ''
            factor = suffix_factor.get(suffix_raw, 1.0)
            add_candidate(value * factor)
        if not candidates:
            for token in re.findall('[-+]?\\d[\\d,]*(?:\\.\\d+)?', raw_text):
                try:
                    add_candidate(float(str(token).replace(',', '')))
                except Exception:
                    continue
        deduped: list[float] = []
        seen = set()
        for num in candidates:
            key = f'{num:.6f}'
            if key in seen:
                continue
            seen.add(key)
            deduped.append(num)
        return deduped

    @staticmethod
    def _value_grounded_in_span(value: str, span: str) -> bool:
        value_norm = re.sub('\\s+', ' ', str(value or '').strip().lower())
        span_norm = re.sub('\\s+', ' ', str(span or '').strip().lower())
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
                    tolerance = max(1e-06, abs(s) * 0.0001)
                    if abs(v - s) <= tolerance:
                        return True
            return False
        value_tokens = [tok for tok in re.findall('[a-z0-9]{4,}', value_norm)]
        if not value_tokens:
            return False
        span_tokens = set(re.findall('[a-z0-9]{4,}', span_norm))
        if not span_tokens:
            return False
        overlap = sum((1 for tok in value_tokens if tok in span_tokens))
        threshold = max(1, int(len(value_tokens) * 0.35))
        return overlap >= threshold

    @staticmethod
    def _contains_numeric_token(text: str) -> bool:
        return re.search('[-+]?\\d[\\d,]*(?:\\.\\d+)?', str(text or '')) is not None

    @staticmethod
    def _source_anchor_keywords(anchor: Optional[str]) -> list[str]:
        normalized = str(anchor or '').strip().lower()
        if not normalized:
            return []
        terms = [normalized]
        for tok in re.split('[^a-z0-9]+', normalized):
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
        normalized = str(anchor or '').strip().lower()
        if not normalized:
            return []
        if normalized == 'cash flow statement':
            return ['cash flow statement', 'statement of cash flows', 'consolidated statement of cash flows', 'cash flows from operating activities', 'cash flows from investing activities', 'cash flows from financing activities', 'net cash provided by operating activities', 'net cash provided by investing activities', 'net cash used in financing activities', 'net increase (decrease) in cash and cash equivalents', 'purchases of property, plant and equipment', 'purchases of pp&e']
        if normalized == 'income statement':
            return ['income statement', 'statement of income', 'statement of operations', 'consolidated statements of operations', 'consolidated statement of income']
        if normalized == 'balance sheet':
            return ['balance sheet', 'statement of financial position', 'consolidated balance sheets', 'consolidated balance sheet']
        return []

    def _is_anchor_grounded_in_citation(self, source_anchor: Optional[str], citation: str, citation_span: str) -> bool:
        markers = self._source_anchor_strict_markers(source_anchor)
        if not markers:
            return True
        blob = f'{citation} {citation_span}'.lower()
        return any((marker in blob for marker in markers))

    @staticmethod
    def _extract_boolean_label(answer: str) -> Optional[str]:
        text = str(answer or '')
        text = re.sub('\\[\\[[^\\]]+\\]\\]', '', text)
        text = re.sub('(?i)@@answer:\\s*', '', text, count=1).strip().lower()
        if re.match('^(yes|no)\\b', text):
            return text.split()[0]
        if re.match('^(true|false)\\b', text):
            return 'yes' if text.startswith('true') else 'no'
        return None

    @staticmethod
    def _extract_year_tokens(text: str) -> list[str]:
        raw = re.findall('(?<!\\d)(?:fy\\s*)?((?:19|20)\\d{2})(?!\\d)', str(text or ''), flags=re.IGNORECASE)
        years: list[str] = []
        seen = set()
        for year in raw:
            if year not in seen:
                seen.add(year)
                years.append(year)
        return years

    @staticmethod
    def _period_granularity(period: Any) -> str:
        text = str(period or '').strip().lower()
        if not text:
            return 'unknown'
        if re.search('\\bq[1-4]\\b', text) or 'quarter' in text or 'three months' in text:
            return 'quarter'
        if 'fy' in text or 'fiscal year' in text or 'year end' in text or re.search('\\b(?:19|20)\\d{2}\\b', text):
            return 'annual'
        return 'unknown'

    @staticmethod
    def _has_quarter_markers(text: str) -> bool:
        blob = str(text or '').lower()
        if not blob:
            return False
        markers = ['three months ended', 'quarter ended', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'first quarter', 'second quarter', 'third quarter', 'fourth quarter']
        return any((marker in blob for marker in markers))

    @staticmethod
    def _has_annual_markers(text: str) -> bool:
        blob = str(text or '').lower()
        if not blob:
            return False
        markers = ['year ended', 'twelve months ended', 'fiscal year', 'annual']
        return any((marker in blob for marker in markers))

    def _value_matches_slot_period(self, *, value: str, slot_period: str, citation: str, citation_span: str) -> bool:
        period_text = str(slot_period or '').strip()
        span_text = str(citation_span or '')
        if not period_text or not span_text:
            return True
        year_aligned = True
        slot_years = set(self._extract_year_tokens(period_text))
        span_lower = span_text.lower()
        if slot_years:
            if any((year in span_lower for year in slot_years)):
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
        if granularity == 'annual':
            has_quarter = self._has_quarter_markers(span_text)
            has_annual = self._has_annual_markers(span_text)
            if has_quarter and (not has_annual):
                return False
        if granularity == 'quarter':
            has_quarter = self._has_quarter_markers(span_text)
            has_annual = self._has_annual_markers(span_text)
            if has_annual and (not has_quarter):
                return False
        return year_aligned

    @staticmethod
    def _extract_query_statement_anchors(query: str) -> list[str]:
        query_lower = str(query or '').strip().lower()
        if not query_lower:
            return []
        anchors: list[str] = []

        def add(anchor: str) -> None:
            if anchor not in anchors:
                anchors.append(anchor)
        balance_markers = ['balance sheet', 'balance sheets', 'statement of financial position']
        income_markers = ['income statement', 'income statements', 'statement of income', 'statement of operations', 'p&l', 'p & l']
        cashflow_markers = ['cash flow statement', 'cash flow statements', 'statement of cash flows']
        if any((marker in query_lower for marker in balance_markers)):
            add('balance sheet')
        if any((marker in query_lower for marker in income_markers)):
            add('income statement')
        if any((marker in query_lower for marker in cashflow_markers)):
            add('cash flow statement')
        return anchors

    @staticmethod
    def _query_has_explicit_statement_anchor(query: str) -> bool:
        return bool(ContextSupport._extract_query_statement_anchors(query))

    def _infer_anchor_for_metric(self, metric: Any) -> Optional[str]:
        metric_key = self._canonical_metric_key(metric)
        if not metric_key:
            return None
        if any((token in metric_key for token in ['capital expenditure', 'capex', 'depreciation', 'amortization', 'dividend', 'cash from operations', 'operating cash flow', 'cash flows from', 'financing activities', 'investing activities'])):
            return 'cash flow statement'
        if any((token in metric_key for token in ['revenue', 'net sales', 'gross profit', 'operating income', 'operating margin', 'net income', 'ebit', 'ebitda', 'cogs', 'cost of goods sold', 'eps', 'earnings per share'])):
            return 'income statement'
        if any((token in metric_key for token in ['total assets', 'total liabilities', 'current assets', 'current liabilities', 'accounts payable', 'accounts receivable', 'inventory', 'property and equipment', 'property, plant and equipment', 'pp&e', 'net ppne', 'net ar', 'quick ratio', 'liability'])):
            return 'balance sheet'
        return None

    def _is_numeric_compute_query(self, query: str, query_state: dict[str, Any]) -> bool:
        if str(query_state.get('answer_type', '')).lower() != 'compute':
            return False
        query_lower = str(query or '').lower()
        metric_key = self._canonical_metric_key(query_state.get('metric', ''))
        if any((marker in query_lower for marker in NUMERIC_QUERY_MARKERS)):
            return True
        return metric_key in NUMERIC_METRIC_KEYS

class ResidualSupport:

    @staticmethod
    def _query_state_prompt_template() -> str:
        return QUERY_STATE_PROMPT

    @staticmethod
    def _query_state_review_prompt_template() -> str:
        return QUERY_STATE_REVIEW_PROMPT

    @staticmethod
    def _agent_execution_prompt_template() -> str:
        return AGENT_EXECUTION_SYSTEM_PROMPT

    @staticmethod
    def _synthesis_prompt_template() -> str:
        return COMPLEX_AGENT_PROMPT_TEMPLATE

    @classmethod
    def _extract_named_entities_from_query(cls, query: str) -> list[str]:
        text = str(query or '').strip()
        if not text:
            return []
        pattern = re.compile("\\b(?:[A-Z][A-Za-z0-9'&.\\-]*)(?:\\s+(?:[A-Z][A-Za-z0-9'&.\\-]*|of|the|and|for|in|on|to|&))*")
        entities: list[str] = []
        seen = set()

        def add_entity(value: str) -> None:
            key = value.lower()
            if not value or key in seen:
                return
            seen.add(key)
            entities.append(value)
        for match in pattern.finditer(text):
            raw = str(match.group(0) or '').strip(' ,?.!;:()[]{}"\'')
            if not raw:
                continue
            words = [word for word in raw.split() if word]
            while words and words[0].lower() in cls._OPEN_DOMAIN_ENTITY_STOPWORDS:
                words = words[1:]
            while words and words[-1].lower() in {'of', 'the', 'and', 'for', 'in', 'on', 'to', '&'}:
                words = words[:-1]
            if not words:
                continue
            normalized = ' '.join(words).strip()
            if not normalized:
                continue
            if ' and ' in normalized.lower():
                parts = [part.strip() for part in re.split('\\band\\b', normalized, flags=re.IGNORECASE) if part.strip()]
                if len(parts) >= 2 and all((len(part.split()) >= 2 for part in parts)):
                    for part in parts:
                        add_entity(part)
                    continue
            if len(words) == 1 and words[0].lower() in cls._OPEN_DOMAIN_ENTITY_STOPWORDS:
                continue
            add_entity(normalized)
        return entities

    @staticmethod
    def _open_domain_relation_hint(query: str) -> str:
        q = str(query or '').lower()
        for phrase in ['government position', 'nationality', 'birthplace', 'spouse', 'occupation', 'portrayed', 'played', 'director', 'author', 'capital']:
            if phrase in q:
                return phrase
        return ''

    @staticmethod
    def _normalize_entities(raw_entities: Any) -> list[str]:
        if isinstance(raw_entities, list):
            return [str(e).strip() for e in raw_entities if str(e).strip()]
        if isinstance(raw_entities, str) and raw_entities.strip():
            return [raw_entities.strip()]
        return []

    @staticmethod
    def _compact_json(data: Any, max_chars: int=1800) -> str:
        return compact_json(data, max_chars=max_chars)

    def _metric_alias_terms(self, metric: str) -> list[str]:
        metric_lower = str(metric or '').strip().lower()
        terms: list[str] = []
        if metric_lower:
            terms.append(metric_lower)
            canonical = self._canonical_metric_key(metric_lower)
            if canonical and canonical != metric_lower:
                terms.append(canonical)
            alias_groups = [(['capex', 'capital expenditure', 'capital expenditures'], ['purchases of property plant and equipment', 'purchases of pp&e', 'additions to property and equipment', 'additions to pp&e']), (['cash from operations', 'operating cash flow'], ['net cash provided by operating activities', 'net cash from operating activities']), (['cash & cash equivalents', 'cash and cash equivalents'], ['cash and cash equivalents']), (['pp&e', 'property and equipment', 'property plant and equipment'], ['property and equipment', 'property plant and equipment', 'property and equipment net', 'property plant and equipment net']), (['fixed asset turnover', 'asset turnover'], ['property and equipment net', 'property and equipment - net', 'property plant and equipment net', 'property plant and equipment - net', 'pp&e', 'net revenue', 'net revenues']), (['revenue', 'net revenue', 'net revenues', 'net sales'], ['revenue', 'net revenues', 'net sales']), (['net income attributable to shareholders', 'net income attributable to shareowners', 'net income attributable to shareowners of the company', 'net income attributable to the company'], ['net income', 'income attributable to shareholders', 'income attributable to shareowners', 'income attributable to shareowners of the company']), (['adjusted eps', 'adjusted earnings per share'], ['adjusted diluted eps', 'non-gaap eps', 'non-gaap earnings per share']), (['total current liabilities', 'current liabilities'], ['total current liabilities', 'current liabilities']), (['capital intensity', 'capital-intensive', 'capital intensive'], ['capital expenditures', 'capex', 'purchases of property plant and equipment', 'property and equipment net', 'fixed assets', 'total assets', 'return on assets']), (['quick ratio'], ['quick ratio', 'quick assets', 'current assets', 'total current assets', 'cash and cash equivalents', 'marketable securities', 'accounts receivable', 'accounts receivable', 'total current liabilities', 'current liabilities']), (['current assets', 'total current assets'], ['current assets', 'total current assets', 'cash and cash equivalents', 'marketable securities', 'accounts receivable', 'inventories', 'prepaids', 'other current assets']), (['current liabilities', 'total current liabilities'], ['current liabilities', 'total current liabilities', 'short-term borrowings', 'accounts payable', 'other current liabilities']), (['segment growth impact', 'segment growth', 'organic growth'], ['worldwide sales change', 'by business segment', 'organic sales', 'acquisitions', 'divestitures', 'total sales change', 'organic sales growth', 'organic local-currency sales', 'consumer segment', 'segment operating performance', 'impact of acquisitions', 'impact of divestitures', 'impact of m&a']), (['debt securities registered to trade', 'debt securities', 'national securities exchange'], ['trading symbol', 'notes due', 'new york stock exchange', 'registered pursuant to section 12(b)', 'section 12(b)', 'title of each class', 'name of each exchange on which registered', 'form 10-q', 'cover page', 'title of each class']), (['operating margin'], ['decrease in gross margin', 'sg&a', 'combat arms earplugs litigation', 'pfas manufacturing', 'exiting russia', 'divestiture-related restructuring', 'special item costs']), (['dividend distribution', 'dividend stability', 'dividend trend'], ['paid dividends since', 'consecutive year of dividend increases', 'cash dividends declared and paid', 'dividend per share'])]
            for triggers, aliases in alias_groups:
                if any((trigger in metric_lower for trigger in triggers)):
                    terms.extend(aliases)
            for tok in re.split('[^a-z0-9]+', metric_lower):
                tok = tok.strip()
                if len(tok) >= 4:
                    terms.append(tok)
        deduped: list[str] = []
        seen = set()
        for term in terms:
            key = term.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(term.strip())
        return deduped

    @staticmethod
    def _is_generic_entity_label(entity: Any) -> bool:
        text = re.sub('\\s+', ' ', str(entity or '').strip().lower())
        if not text:
            return True
        generic = {'company', 'entity', 'firm', 'business', 'organization', 'corporation', 'ceo', 'management', 'executive', 'leadership'}
        return text in generic

    def _is_capex_metric(self, metric: Any) -> bool:
        key = self._canonical_metric_key(metric)
        return 'capital expenditure' in key or 'capex' in key

    def _metric_expects_ratio_value(self, metric: Any) -> bool:
        key = self._canonical_metric_key(metric)
        if not key:
            return False
        ratio_markers = ['ratio', 'margin', 'percent', '%', 'per cent', 'as a %', 'as a percent', 'year-over-year', 'yoy']
        return any((marker in key for marker in ratio_markers))

    def _is_dividend_metric(self, metric: Any) -> bool:
        key = self._canonical_metric_key(metric)
        return 'dividend' in key

    @staticmethod
    def _positive_amount_string(value: str) -> str:
        text = str(value or '')
        if not text.strip():
            return text
        paren_match = re.search('\\(\\s*(\\d[\\d,]*(?:\\.\\d+)?)\\s*\\)', text)
        if paren_match:
            start, end = paren_match.span(0)
            return text[:start] + paren_match.group(1) + text[end:]
        minus_match = re.search('(^|\\s)-\\s*(\\d[\\d,]*(?:\\.\\d+)?)', text)
        if minus_match:
            start, end = minus_match.span(0)
            lead = minus_match.group(1)
            number = minus_match.group(2)
            return text[:start] + f'{lead}{number}' + text[end:]
        return text

    @staticmethod
    def _validate_context_atomization_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return (False, 'top-level must be JSON object')
        atoms = data.get('atoms')
        if not isinstance(atoms, list):
            return (False, 'atoms must be JSON array')
        for idx, item in enumerate(atoms):
            if not isinstance(item, dict):
                return (False, f'atoms[{idx}] must be JSON object')
            atom_id = item.get('atom_id')
            citation = item.get('citation')
            span = item.get('span')
            supports_slots = item.get('supports_slots', [])
            if not isinstance(atom_id, str) or not atom_id.strip():
                return (False, f'atoms[{idx}].atom_id must be non-empty string')
            if not isinstance(citation, str) or not citation.strip():
                return (False, f'atoms[{idx}].citation must be non-empty string')
            if CITATION_RE.search(citation) is None:
                return (False, f'atoms[{idx}].citation must match [[Title, Page X, Chunk Y]]')
            if not isinstance(span, str) or not span.strip():
                return (False, f'atoms[{idx}].span must be non-empty string')
            if not isinstance(supports_slots, list):
                return (False, f'atoms[{idx}].supports_slots must be JSON array')
            for s_idx, slot in enumerate(supports_slots):
                if not isinstance(slot, (str, dict)):
                    return (False, f'atoms[{idx}].supports_slots[{s_idx}] must be string or object')
        return (True, '')

    def _context_atomization_retry_message(self, failed_output: Any, reason: str) -> str:
        return CONTEXT_ATOMIZATION_RETRY_PROMPT.format(error=reason, previous_output=self._compact_json(failed_output, max_chars=900))

    def _normalize_atoms(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        atoms_raw = data.get('atoms', [])
        if not isinstance(atoms_raw, list):
            return []
        atoms: list[dict[str, Any]] = []
        seen_ids = set()
        for idx, item in enumerate(atoms_raw, start=1):
            if not isinstance(item, dict):
                continue
            atom_id = str(item.get('atom_id', '') or f'a{idx}').strip() or f'a{idx}'
            if atom_id in seen_ids:
                continue
            citation = str(item.get('citation', '') or '').strip()
            span = str(item.get('span', '') or '').strip()
            if CITATION_RE.search(citation) is None:
                continue
            if not span:
                continue
            supports_slots_raw = item.get('supports_slots', [])
            supports_slots: list[str] = []
            if isinstance(supports_slots_raw, list):
                for slot in supports_slots_raw:
                    struct = self._parse_slot_struct(slot)
                    if struct:
                        supports_slots.append(self._normalize_slot(struct))
                        continue
                    text = str(slot or '').strip()
                    if text:
                        supports_slots.append(text)
            atoms.append({'atom_id': atom_id, 'citation': citation, 'span': span[:480], 'supports_slots': supports_slots})
            seen_ids.add(atom_id)
        return atoms

    @staticmethod
    def _validate_context_packing_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return (False, 'top-level must be JSON object')
        selected = data.get('selected_atom_ids')
        if not isinstance(selected, list) or any((not isinstance(x, str) for x in selected)):
            return (False, 'selected_atom_ids must be string array')
        slot_coverage = data.get('slot_coverage')
        if not isinstance(slot_coverage, dict):
            return (False, 'slot_coverage must be object')
        missing_slots = data.get('missing_slots')
        if not isinstance(missing_slots, list):
            return (False, 'missing_slots must be JSON array')
        return (True, '')

    def _context_packing_retry_message(self, failed_output: Any, reason: str) -> str:
        return CONTEXT_PACKING_RETRY_PROMPT.format(error=reason, previous_output=self._compact_json(failed_output, max_chars=900))

    async def _extract_context_atoms(self, query_state: dict[str, Any], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized = self._serialize_nodes(nodes, query_state=query_state, max_text_chars=760)
        if not serialized.strip():
            return []
        base_messages = [{'role': 'user', 'content': CONTEXT_ATOMIZATION_PROMPT.format(query_state=self._compact_json(query_state, max_chars=1000), context=serialized)}, {'role': 'user', 'content': CONTEXT_ATOMIZATION_FORMAT_INSTRUCTION}]
        data, ok, attempts = await generate_json_with_retries(self.llm, base_messages, self._validate_context_atomization_json, self._context_atomization_retry_message, max_attempts=3, logger=logger, warning_prefix='context atomization json generation failed', model=self.stage_model)
        if not ok and attempts:
            logger.warning('Context atomization failed schema validation after %d attempts', len(attempts))
        return self._normalize_atoms(data)

    async def _pack_context_atoms(self, query_state: dict[str, Any], atoms: list[dict[str, Any]], budget_chars: int) -> dict[str, Any]:
        required = self._required_slots(query_state)
        empty_result = {'selected_atom_ids': [], 'slot_coverage': {}, 'missing_slots': required, 'compressed_context': ''}
        if not atoms:
            return empty_result
        base_messages = [{'role': 'user', 'content': CONTEXT_PACKING_PROMPT.format(query_state=self._compact_json(query_state, max_chars=1000), budget_chars=int(max(800, budget_chars)), atoms=self._compact_json({'atoms': atoms}, max_chars=5000))}, {'role': 'user', 'content': CONTEXT_PACKING_FORMAT_INSTRUCTION}]
        data, ok, attempts = await generate_json_with_retries(self.llm, base_messages, self._validate_context_packing_json, self._context_packing_retry_message, max_attempts=3, logger=logger, warning_prefix='context packing json generation failed', model=self.stage_model)
        if not ok and attempts:
            logger.warning('Context packing failed schema validation after %d attempts', len(attempts))
        selected_raw = data.get('selected_atom_ids', [])
        selected_ids: list[str] = []
        if isinstance(selected_raw, list):
            selected_ids = [str(x).strip() for x in selected_raw if str(x).strip()]
        slot_coverage = data.get('slot_coverage', {})
        if not isinstance(slot_coverage, dict):
            slot_coverage = {}
        model_missing_slots = self._sanitize_missing_slots(query_state, data.get('missing_slots', None))
        if model_missing_slots is None:
            covered = set()
            required_map = {self._normalize_slot(slot): slot for slot in required}
            for slot_name in slot_coverage.keys():
                key = self._normalize_slot(slot_name)
                if key in required_map:
                    covered.add(key)
            model_missing_slots = [slot for slot in required if self._normalize_slot(slot) not in covered]
        valid_atom_ids = {str(atom.get('atom_id', '') or '').strip() for atom in atoms if str(atom.get('atom_id', '') or '').strip()}
        selected_set = {atom_id for atom_id in selected_ids if atom_id in valid_atom_ids}
        blocks: list[str] = []
        total_chars = 0
        selected_atom_ids: list[str] = []
        for atom in atoms:
            atom_id = str(atom.get('atom_id', '') or '')
            if atom_id not in selected_set:
                continue
            block = f"{atom.get('citation', '')}\n{atom.get('span', '')}".strip()
            if not block:
                continue
            if total_chars + len(block) > budget_chars and blocks:
                break
            selected_atom_ids.append(atom_id)
            blocks.append(block)
            total_chars += len(block) + 2
        if selected_atom_ids or not required:
            return {'selected_atom_ids': selected_atom_ids, 'slot_coverage': slot_coverage, 'missing_slots': model_missing_slots, 'compressed_context': '\n\n'.join(blocks)}
        return empty_result

    async def _refresh_context_and_slots(self, state, trace_step: str) -> None:
        nodes = self._dedupe_nodes(state.all_context_data, max_nodes=self.context_node_budget)
        atoms = await self._extract_context_atoms(state.query_state, nodes)
        packed = await self._pack_context_atoms(state.query_state, atoms, budget_chars=self.context_char_budget)
        state.evidence_atoms = atoms
        compressed = str(packed.get('compressed_context', '') or '').strip()
        fallback_context = self._build_context_excerpt(nodes, limit=8, query_state=state.query_state)
        use_compressed = bool(compressed)
        if use_compressed and state.missing_slots and (len(compressed) < 220):
            use_compressed = False
        if use_compressed and self._query_has_explicit_statement_anchor(state.user_query) and (len(compressed) < 360):
            use_compressed = False
        if use_compressed and state.evidence_ledger:
            compressed_citations = set(self._context_citation_map(compressed).keys())
            ledger_citations = {self._normalize_citation(entry.get('citation', '')) for entry in state.evidence_ledger if self._normalize_citation(entry.get('citation', ''))}
            if ledger_citations and compressed_citations.isdisjoint(ledger_citations):
                use_compressed = False
        state.context = compressed if use_compressed else fallback_context
        if len(state.context) > self.context_char_budget:
            state.context = self._extract_relevant_span(state.context, query_state=state.query_state, max_chars=self.context_char_budget)
        state.missing_slots = self._resolve_missing_slots(state.query_state, state.evidence_ledger, model_missing_slots=packed.get('missing_slots', None), trust_model_missing=False)
        append_trace(state.trace, step=trace_step, input={'nodes': len(nodes), 'ledger_entries': len(state.evidence_ledger), 'required_slots': self._required_slots(state.query_state)}, output={'atoms': len(atoms), 'selected_atom_ids': packed.get('selected_atom_ids', []), 'missing_slots': state.missing_slots, 'context_chars': len(state.context)})
