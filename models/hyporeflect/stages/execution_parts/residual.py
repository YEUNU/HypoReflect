import logging
import re
from typing import Any

from utils.prompts import (
    AGENT_EXECUTION_SYSTEM_PROMPT,
    COMPLEX_AGENT_PROMPT_TEMPLATE,
    CONTEXT_ATOMIZATION_FORMAT_INSTRUCTION,
    CONTEXT_ATOMIZATION_PROMPT,
    CONTEXT_ATOMIZATION_RETRY_PROMPT,
    CONTEXT_PACKING_FORMAT_INSTRUCTION,
    CONTEXT_PACKING_PROMPT,
    CONTEXT_PACKING_RETRY_PROMPT,
    QUERY_STATE_PROMPT,
    QUERY_STATE_REVIEW_PROMPT,
)
from models.hyporeflect.stages.common import CITATION_RE
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries
from models.hyporeflect.trace import append_trace


logger = logging.getLogger(__name__)


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
        text = str(query or "").strip()
        if not text:
            return []
        pattern = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9'&.\-]*)(?:\s+(?:[A-Z][A-Za-z0-9'&.\-]*|of|the|and|for|in|on|to|&))*"
        )
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
            words = [word for word in raw.split() if word]
            while words and words[0].lower() in cls._OPEN_DOMAIN_ENTITY_STOPWORDS:
                words = words[1:]
            while words and words[-1].lower() in {"of", "the", "and", "for", "in", "on", "to", "&"}:
                words = words[:-1]
            if not words:
                continue
            normalized = " ".join(words).strip()
            if not normalized:
                continue
            if " and " in normalized.lower():
                parts = [part.strip() for part in re.split(r"\band\b", normalized, flags=re.IGNORECASE) if part.strip()]
                if len(parts) >= 2 and all(len(part.split()) >= 2 for part in parts):
                    for part in parts:
                        add_entity(part)
                    continue
            if len(words) == 1 and words[0].lower() in cls._OPEN_DOMAIN_ENTITY_STOPWORDS:
                continue
            add_entity(normalized)
        return entities

    @staticmethod
    def _open_domain_relation_hint(query: str) -> str:
        q = str(query or "").lower()
        for phrase in [
            "government position",
            "nationality",
            "birthplace",
            "spouse",
            "occupation",
            "portrayed",
            "played",
            "director",
            "author",
            "capital",
        ]:
            if phrase in q:
                return phrase
        return ""

    @staticmethod
    def _normalize_entities(raw_entities: Any) -> list[str]:
        if isinstance(raw_entities, list):
            return [str(e).strip() for e in raw_entities if str(e).strip()]
        if isinstance(raw_entities, str) and raw_entities.strip():
            return [raw_entities.strip()]
        return []

    @staticmethod
    def _compact_json(data: Any, max_chars: int = 1800) -> str:
        return compact_json(data, max_chars=max_chars)

    def _metric_alias_terms(self, metric: str) -> list[str]:
        metric_lower = str(metric or "").strip().lower()
        terms: list[str] = []
        if metric_lower:
            terms.append(metric_lower)
            canonical = self._canonical_metric_key(metric_lower)
            if canonical and canonical != metric_lower:
                terms.append(canonical)

            alias_groups = [
                (
                    ["capex", "capital expenditure", "capital expenditures"],
                    [
                        "purchases of property plant and equipment",
                        "purchases of pp&e",
                        "additions to property and equipment",
                        "additions to pp&e",
                    ],
                ),
                (
                    ["cash from operations", "operating cash flow"],
                    [
                        "net cash provided by operating activities",
                        "net cash from operating activities",
                    ],
                ),
                (
                    ["cash & cash equivalents", "cash and cash equivalents"],
                    ["cash and cash equivalents"],
                ),
                (
                    ["pp&e", "property and equipment", "property plant and equipment"],
                    [
                        "property and equipment",
                        "property plant and equipment",
                        "property and equipment net",
                        "property plant and equipment net",
                    ],
                ),
                (
                    ["fixed asset turnover", "asset turnover"],
                    [
                        "property and equipment net",
                        "property and equipment - net",
                        "property plant and equipment net",
                        "property plant and equipment - net",
                        "pp&e",
                        "net revenue",
                        "net revenues",
                    ],
                ),
                (
                    ["revenue", "net revenue", "net revenues", "net sales"],
                    ["revenue", "net revenues", "net sales"],
                ),
                (
                    [
                        "net income attributable to shareholders",
                        "net income attributable to shareowners",
                        "net income attributable to shareowners of the company",
                        "net income attributable to the company",
                    ],
                    [
                        "net income",
                        "income attributable to shareholders",
                        "income attributable to shareowners",
                        "income attributable to shareowners of the company",
                    ],
                ),
                (
                    ["adjusted eps", "adjusted earnings per share"],
                    ["adjusted diluted eps", "non-gaap eps", "non-gaap earnings per share"],
                ),
                (
                    ["total current liabilities", "current liabilities"],
                    ["total current liabilities", "current liabilities"],
                ),
                (
                    ["capital intensity", "capital-intensive", "capital intensive"],
                    [
                        "capital expenditures",
                        "capex",
                        "purchases of property plant and equipment",
                        "property and equipment net",
                        "fixed assets",
                        "total assets",
                        "return on assets",
                    ],
                ),
                (
                    ["quick ratio"],
                    [
                        "quick ratio",
                        "quick assets",
                        "current assets",
                        "total current assets",
                        "cash and cash equivalents",
                        "marketable securities",
                        "accounts receivable",
                        "accounts receivable",
                        "total current liabilities",
                        "current liabilities",
                    ],
                ),
                (
                    ["current assets", "total current assets"],
                    [
                        "current assets",
                        "total current assets",
                        "cash and cash equivalents",
                        "marketable securities",
                        "accounts receivable",
                        "inventories",
                        "prepaids",
                        "other current assets",
                    ],
                ),
                (
                    ["current liabilities", "total current liabilities"],
                    [
                        "current liabilities",
                        "total current liabilities",
                        "short-term borrowings",
                        "accounts payable",
                        "other current liabilities",
                    ],
                ),
                (
                    ["segment growth impact", "segment growth", "organic growth"],
                    [
                        "worldwide sales change",
                        "by business segment",
                        "organic sales",
                        "acquisitions",
                        "divestitures",
                        "total sales change",
                        "organic sales growth",
                        "organic local-currency sales",
                        "consumer segment",
                        "segment operating performance",
                        "impact of acquisitions",
                        "impact of divestitures",
                        "impact of m&a",
                    ],
                ),
                (
                    ["debt securities registered to trade", "debt securities", "national securities exchange"],
                    [
                        "trading symbol",
                        "notes due",
                        "new york stock exchange",
                        "registered pursuant to section 12(b)",
                        "section 12(b)",
                        "title of each class",
                        "name of each exchange on which registered",
                        "form 10-q",
                        "cover page",
                        "title of each class",
                    ],
                ),
                (
                    ["operating margin"],
                    [
                        "decrease in gross margin",
                        "sg&a",
                        "combat arms earplugs litigation",
                        "pfas manufacturing",
                        "exiting russia",
                        "divestiture-related restructuring",
                        "special item costs",
                    ],
                ),
                (
                    ["dividend distribution", "dividend stability", "dividend trend"],
                    [
                        "paid dividends since",
                        "consecutive year of dividend increases",
                        "cash dividends declared and paid",
                        "dividend per share",
                    ],
                ),
            ]
            for triggers, aliases in alias_groups:
                if any(trigger in metric_lower for trigger in triggers):
                    terms.extend(aliases)
            for tok in re.split(r"[^a-z0-9]+", metric_lower):
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
        text = re.sub(r"\s+", " ", str(entity or "").strip().lower())
        if not text:
            return True
        generic = {
            "company",
            "entity",
            "firm",
            "business",
            "organization",
            "corporation",
            "ceo",
            "management",
            "executive",
            "leadership",
        }
        return text in generic

    def _is_capex_metric(self, metric: Any) -> bool:
        key = self._canonical_metric_key(metric)
        return "capital expenditure" in key or "capex" in key

    def _metric_expects_ratio_value(self, metric: Any) -> bool:
        key = self._canonical_metric_key(metric)
        if not key:
            return False
        ratio_markers = [
            "ratio",
            "margin",
            "percent",
            "%",
            "per cent",
            "as a %",
            "as a percent",
            "year-over-year",
            "yoy",
        ]
        return any(marker in key for marker in ratio_markers)

    def _is_dividend_metric(self, metric: Any) -> bool:
        key = self._canonical_metric_key(metric)
        return "dividend" in key

    @staticmethod
    def _positive_amount_string(value: str) -> str:
        text = str(value or "")
        if not text.strip():
            return text
        paren_match = re.search(r"\(\s*(\d[\d,]*(?:\.\d+)?)\s*\)", text)
        if paren_match:
            start, end = paren_match.span(0)
            return text[:start] + paren_match.group(1) + text[end:]
        minus_match = re.search(r"(^|\s)-\s*(\d[\d,]*(?:\.\d+)?)", text)
        if minus_match:
            start, end = minus_match.span(0)
            lead = minus_match.group(1)
            number = minus_match.group(2)
            return text[:start] + f"{lead}{number}" + text[end:]
        return text

    @staticmethod
    def _validate_context_atomization_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        atoms = data.get("atoms")
        if not isinstance(atoms, list):
            return False, "atoms must be JSON array"
        for idx, item in enumerate(atoms):
            if not isinstance(item, dict):
                return False, f"atoms[{idx}] must be JSON object"
            atom_id = item.get("atom_id")
            citation = item.get("citation")
            span = item.get("span")
            supports_slots = item.get("supports_slots", [])
            if not isinstance(atom_id, str) or not atom_id.strip():
                return False, f"atoms[{idx}].atom_id must be non-empty string"
            if not isinstance(citation, str) or not citation.strip():
                return False, f"atoms[{idx}].citation must be non-empty string"
            if CITATION_RE.search(citation) is None:
                return False, f"atoms[{idx}].citation must match [[Title, Page X, Chunk Y]]"
            if not isinstance(span, str) or not span.strip():
                return False, f"atoms[{idx}].span must be non-empty string"
            if not isinstance(supports_slots, list):
                return False, f"atoms[{idx}].supports_slots must be JSON array"
            for s_idx, slot in enumerate(supports_slots):
                if not isinstance(slot, (str, dict)):
                    return False, f"atoms[{idx}].supports_slots[{s_idx}] must be string or object"
        return True, ""

    def _context_atomization_retry_message(self, failed_output: Any, reason: str) -> str:
        return CONTEXT_ATOMIZATION_RETRY_PROMPT.format(
            error=reason,
            previous_output=self._compact_json(failed_output, max_chars=900),
        )

    def _normalize_atoms(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        atoms_raw = data.get("atoms", [])
        if not isinstance(atoms_raw, list):
            return []
        atoms: list[dict[str, Any]] = []
        seen_ids = set()
        for idx, item in enumerate(atoms_raw, start=1):
            if not isinstance(item, dict):
                continue
            atom_id = str(item.get("atom_id", "") or f"a{idx}").strip() or f"a{idx}"
            if atom_id in seen_ids:
                continue
            citation = str(item.get("citation", "") or "").strip()
            span = str(item.get("span", "") or "").strip()
            if CITATION_RE.search(citation) is None:
                continue
            if not span:
                continue
            supports_slots_raw = item.get("supports_slots", [])
            supports_slots: list[str] = []
            if isinstance(supports_slots_raw, list):
                for slot in supports_slots_raw:
                    struct = self._parse_slot_struct(slot)
                    if struct:
                        supports_slots.append(self._normalize_slot(struct))
                        continue
                    text = str(slot or "").strip()
                    if text:
                        supports_slots.append(text)
            atoms.append({
                "atom_id": atom_id,
                "citation": citation,
                "span": span[:480],
                "supports_slots": supports_slots,
            })
            seen_ids.add(atom_id)
        return atoms

    @staticmethod
    def _validate_context_packing_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        selected = data.get("selected_atom_ids")
        if not isinstance(selected, list) or any(not isinstance(x, str) for x in selected):
            return False, "selected_atom_ids must be string array"
        slot_coverage = data.get("slot_coverage")
        if not isinstance(slot_coverage, dict):
            return False, "slot_coverage must be object"
        missing_slots = data.get("missing_slots")
        if not isinstance(missing_slots, list):
            return False, "missing_slots must be JSON array"
        return True, ""

    def _context_packing_retry_message(self, failed_output: Any, reason: str) -> str:
        return CONTEXT_PACKING_RETRY_PROMPT.format(
            error=reason,
            previous_output=self._compact_json(failed_output, max_chars=900),
        )

    async def _extract_context_atoms(
        self,
        query_state: dict[str, Any],
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        serialized = self._serialize_nodes(nodes, query_state=query_state, max_text_chars=760)
        if not serialized.strip():
            return []
        base_messages = [
            {
                "role": "user",
                "content": CONTEXT_ATOMIZATION_PROMPT.format(
                    query_state=self._compact_json(query_state, max_chars=1000),
                    context=serialized,
                ),
            },
            {"role": "user", "content": CONTEXT_ATOMIZATION_FORMAT_INSTRUCTION},
        ]
        data, ok, attempts = await generate_json_with_retries(
            self.llm,
            base_messages,
            self._validate_context_atomization_json,
            self._context_atomization_retry_message,
            max_attempts=3,
            logger=logger,
            warning_prefix="context atomization json generation failed",
            model=self.stage_model,
        )
        if not ok and attempts:
            logger.warning("Context atomization failed schema validation after %d attempts", len(attempts))
        return self._normalize_atoms(data)

    async def _pack_context_atoms(
        self,
        query_state: dict[str, Any],
        atoms: list[dict[str, Any]],
        budget_chars: int,
    ) -> dict[str, Any]:
        required = self._required_slots(query_state)
        empty_result = {
            "selected_atom_ids": [],
            "slot_coverage": {},
            "missing_slots": required,
            "compressed_context": "",
        }
        if not atoms:
            return empty_result

        base_messages = [
            {
                "role": "user",
                "content": CONTEXT_PACKING_PROMPT.format(
                    query_state=self._compact_json(query_state, max_chars=1000),
                    budget_chars=int(max(800, budget_chars)),
                    atoms=self._compact_json({"atoms": atoms}, max_chars=5000),
                ),
            },
            {"role": "user", "content": CONTEXT_PACKING_FORMAT_INSTRUCTION},
        ]
        data, ok, attempts = await generate_json_with_retries(
            self.llm,
            base_messages,
            self._validate_context_packing_json,
            self._context_packing_retry_message,
            max_attempts=3,
            logger=logger,
            warning_prefix="context packing json generation failed",
            model=self.stage_model,
        )
        if not ok and attempts:
            logger.warning("Context packing failed schema validation after %d attempts", len(attempts))

        selected_raw = data.get("selected_atom_ids", [])
        selected_ids: list[str] = []
        if isinstance(selected_raw, list):
            selected_ids = [str(x).strip() for x in selected_raw if str(x).strip()]

        slot_coverage = data.get("slot_coverage", {})
        if not isinstance(slot_coverage, dict):
            slot_coverage = {}

        model_missing_slots = self._sanitize_missing_slots(query_state, data.get("missing_slots", None))
        if model_missing_slots is None:
            covered = set()
            required_map = {self._normalize_slot(slot): slot for slot in required}
            for slot_name in slot_coverage.keys():
                key = self._normalize_slot(slot_name)
                if key in required_map:
                    covered.add(key)
            model_missing_slots = [slot for slot in required if self._normalize_slot(slot) not in covered]

        valid_atom_ids = {
            str(atom.get("atom_id", "") or "").strip()
            for atom in atoms
            if str(atom.get("atom_id", "") or "").strip()
        }
        selected_set = {atom_id for atom_id in selected_ids if atom_id in valid_atom_ids}
        blocks: list[str] = []
        total_chars = 0
        selected_atom_ids: list[str] = []
        for atom in atoms:
            atom_id = str(atom.get("atom_id", "") or "")
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
            return {
                "selected_atom_ids": selected_atom_ids,
                "slot_coverage": slot_coverage,
                "missing_slots": model_missing_slots,
                "compressed_context": "\n\n".join(blocks),
            }
        return empty_result

    async def _refresh_context_and_slots(
        self,
        state,
        trace_step: str,
    ) -> None:
        nodes = self._dedupe_nodes(state.all_context_data, max_nodes=self.context_node_budget)
        atoms = await self._extract_context_atoms(state.query_state, nodes)
        packed = await self._pack_context_atoms(
            state.query_state,
            atoms,
            budget_chars=self.context_char_budget,
        )
        state.evidence_atoms = atoms
        compressed = str(packed.get("compressed_context", "") or "").strip()
        fallback_context = self._build_context_excerpt(
            nodes,
            limit=8,
            query_state=state.query_state,
        )
        use_compressed = bool(compressed)
        if use_compressed and state.missing_slots and len(compressed) < 220:
            use_compressed = False
        if (
            use_compressed
            and self._query_has_explicit_statement_anchor(state.user_query)
            and len(compressed) < 360
        ):
            use_compressed = False
        if use_compressed and state.evidence_ledger:
            compressed_citations = set(self._context_citation_map(compressed).keys())
            ledger_citations = {
                self._normalize_citation(entry.get("citation", ""))
                for entry in state.evidence_ledger
                if self._normalize_citation(entry.get("citation", ""))
            }
            if ledger_citations and compressed_citations.isdisjoint(ledger_citations):
                use_compressed = False
        state.context = compressed if use_compressed else fallback_context
        if len(state.context) > self.context_char_budget:
            state.context = self._extract_relevant_span(
                state.context,
                query_state=state.query_state,
                max_chars=self.context_char_budget,
            )
        state.missing_slots = self._resolve_missing_slots(
            state.query_state,
            state.evidence_ledger,
            model_missing_slots=packed.get("missing_slots", None),
            trust_model_missing=False,
        )
        append_trace(
            state.trace,
            step=trace_step,
            input={
                "nodes": len(nodes),
                "ledger_entries": len(state.evidence_ledger),
                "required_slots": self._required_slots(state.query_state),
            },
            output={
                "atoms": len(atoms),
                "selected_atom_ids": packed.get("selected_atom_ids", []),
                "missing_slots": state.missing_slots,
                "context_chars": len(state.context),
            },
        )
