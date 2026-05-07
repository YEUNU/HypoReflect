import ast
import logging
import math
import operator
import re
from typing import Any, Optional

from utils.prompts import CALCULATION_PLAN_PROMPT, CALCULATION_PLAN_RETRY_PROMPT
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.common import extract_first_number
from models.hyporeflect.stages.llm_json import compact_json, generate_json_with_retries


logger = logging.getLogger(__name__)


class CalculatorSupport:
    @staticmethod
    def _safe_eval_arithmetic(expression: str) -> float:
        allowed_binops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        allowed_unary = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

        def eval_node(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("calculator only accepts numeric constants")
            if isinstance(node, ast.Num):  # pragma: no cover (py<3.8 compatibility)
                return float(node.n)
            if isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in allowed_binops:
                    raise ValueError("unsupported arithmetic operator")
                left = eval_node(node.left)
                right = eval_node(node.right)
                if op_type is ast.Div and right == 0:
                    raise ValueError("division by zero")
                value = allowed_binops[op_type](left, right)
                if not (-1e18 <= value <= 1e18):
                    raise ValueError("calculator result out of range")
                return float(value)
            if isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in allowed_unary:
                    raise ValueError("unsupported unary operator")
                value = allowed_unary[op_type](eval_node(node.operand))
                if not (-1e18 <= value <= 1e18):
                    raise ValueError("calculator result out of range")
                return float(value)
            raise ValueError("invalid arithmetic expression")

        parsed = ast.parse(expression, mode="eval")
        return eval_node(parsed)

    def _call_calculator(self, expression: Any, precision: Any = None) -> dict[str, Any]:
        expr = str(expression or "").strip()
        if not expr:
            return {"ok": False, "error": "empty expression"}
        expr = expr.replace(",", "").replace("$", "").replace("%", "")
        expr = re.sub(r"\s+", " ", expr).strip()

        try:
            value = self._safe_eval_arithmetic(expr)
        except Exception as e:
            return {"ok": False, "error": str(e), "expression": expr}

        precision_val: Optional[int] = None
        if precision is not None and str(precision).strip() != "":
            try:
                precision_val = int(precision)
                precision_val = max(0, min(precision_val, 8))
            except Exception:
                precision_val = None

        if precision_val is not None:
            value = round(value, precision_val)
            result = f"{value:.{precision_val}f}"
        else:
            result = f"{value:.12g}"

        return {
            "ok": True,
            "expression": expr,
            "precision": precision_val,
            "result": result,
            "result_numeric": value,
        }

    @staticmethod
    def _extract_primary_financial_number(text: str) -> Optional[float]:
        raw = str(text or "")
        if not raw.strip():
            return None

        def parse_token(token: str) -> Optional[float]:
            cleaned = str(token).strip()
            if not cleaned:
                return None
            negative = False
            if cleaned.startswith("(") and cleaned.endswith(")"):
                negative = True
                cleaned = cleaned[1:-1]
            cleaned = cleaned.replace("$", "").replace(",", "").replace("%", "").strip()
            if cleaned.startswith("+"):
                cleaned = cleaned[1:].strip()
            if not cleaned:
                return None
            try:
                value = float(cleaned)
            except Exception:
                return None
            if negative:
                value = -value
            return value

        # Prefer explicit monetary tokens first.
        money_tokens = re.findall(r"\$\s*\(?\d[\d,]*(?:\.\d+)?\)?", raw)
        for token in money_tokens:
            value = parse_token(token)
            if value is not None:
                return value

        # Then prefer accounting parenthesized numeric tokens.
        paren_tokens = re.findall(r"\(\s*\d[\d,]*(?:\.\d+)?\s*\)", raw)
        for token in paren_tokens:
            value = parse_token(token)
            if value is not None:
                return value

        # Fallback: first non-year numeric token.
        for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", raw):
            value = parse_token(token)
            if value is None:
                continue
            abs_value = abs(value)
            if float(abs_value).is_integer() and 1900 <= abs_value <= 2100:
                continue
            return value
        return None

    @staticmethod
    def _extract_numeric_literals(text: str) -> list[float]:
        values: list[float] = []
        for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text or "")):
            try:
                values.append(float(str(match).replace(",", "")))
            except Exception:
                continue
        return values

    @staticmethod
    def _extract_largest_non_year_number(text: str) -> Optional[float]:
        candidates: list[float] = []
        for value in CalculatorSupport._extract_numeric_literals(text):
            abs_value = abs(float(value))
            if float(abs_value).is_integer() and 1900 <= abs_value <= 2100:
                continue
            candidates.append(float(value))
        if not candidates:
            return None
        return max(candidates, key=lambda x: abs(x))

    def _expression_ledger_match_count(
        self,
        expression: str,
        evidence_ledger: list[dict[str, str]],
    ) -> int:
        expr_numbers = self._extract_numeric_literals(expression)
        if not expr_numbers:
            return 0
        ledger_numbers: list[float] = []
        for entry in evidence_ledger:
            ledger_numbers.extend(self._extract_numeric_literals(str(entry.get("value", "") or "")))
        if not ledger_numbers:
            return 0

        used_ledger_idx = set()
        match_count = 0
        for expr_value in expr_numbers:
            matched_idx = None
            for idx, ledger_value in enumerate(ledger_numbers):
                if idx in used_ledger_idx:
                    continue
                tolerance = max(1e-6, abs(ledger_value) * 1e-4)
                if abs(expr_value - ledger_value) <= tolerance:
                    matched_idx = idx
                    break
            if matched_idx is None:
                continue
            used_ledger_idx.add(matched_idx)
            match_count += 1
        return match_count

    @staticmethod
    def _format_numeric_for_slot_value(value: float) -> str:
        if not math.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.8f}".rstrip("0").rstrip(".")

    def _deterministic_compute_slot_entries(
        self,
        query_state: dict[str, Any],
        missing_slots: list[Any],
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        if str(query_state.get("answer_type", "")).strip().lower() != "compute":
            return []
        if not isinstance(missing_slots, list) or not missing_slots:
            return []
        if not isinstance(nodes, list) or not nodes:
            return []

        required = self._required_slots(query_state)
        required_map = {self._normalize_slot(slot): slot for slot in required}
        slot_keys_to_fill = [
            key
            for key in (self._normalize_slot(slot) for slot in missing_slots)
            if key and key in required_map
        ]
        if not slot_keys_to_fill:
            return []

        entries: list[dict[str, str]] = []
        seen_slot = set()
        for slot_key in slot_keys_to_fill:
            if slot_key in seen_slot:
                continue
            seen_slot.add(slot_key)

            slot_raw = required_map[slot_key]
            slot_struct = self._parse_slot_struct(slot_raw)
            if not slot_struct:
                continue

            slot_metric = str(slot_struct.get("metric", "") or "").strip()
            metric_terms = [
                str(term).strip().lower()
                for term in self._metric_alias_terms(slot_metric)
                if len(str(term).strip()) >= 4
            ][:16]
            slot_years = set(self._extract_year_tokens(str(slot_struct.get("period", "") or "")))
            slot_anchor = str(slot_struct.get("source_anchor", "") or "").strip().lower()
            is_capex_cashflow_slot = self._is_capex_metric(slot_metric) and slot_anchor == "cash flow statement"
            capex_amount_markers = [
                "capital expenditure",
                "capital expenditures",
                "purchases of property",
                "purchases of pp&e",
                "additions to property and equipment",
                "additions to pp&e",
                "property, plant and equipment",
                "property and equipment",
            ]
            capex_ratio_markers = [
                "as a percentage",
                "% of net revenues",
                "% of net revenue",
                "% of revenue",
                "percent of net revenues",
                "percent of revenue",
            ]

            best_value = ""
            best_citation = ""
            best_score = -1

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                text = re.sub(r"\s+", " ", str(node.get("text", "") or "").strip())
                if not text:
                    continue
                citation = self._node_citation(node)
                citation_title = str(node.get("title") or node.get("doc") or "")
                lower = text.lower()

                if is_capex_cashflow_slot:
                    if any(marker in lower for marker in capex_ratio_markers):
                        continue
                    if not any(marker in lower for marker in capex_amount_markers):
                        continue

                metric_hit_count = sum(1 for term in metric_terms if term and term in lower)
                if metric_terms and metric_hit_count == 0:
                    continue

                span_years = set(self._extract_year_tokens(text))
                title_years = set(self._extract_year_tokens(citation_title))
                year_overlap_span = bool(slot_years and span_years and not slot_years.isdisjoint(span_years))
                year_overlap_title = bool(slot_years and title_years and not slot_years.isdisjoint(title_years))
                if slot_years and span_years and not year_overlap_span:
                    continue
                if slot_years and not span_years and title_years and not year_overlap_title:
                    continue

                anchor_ok = True
                if slot_anchor:
                    anchor_ok = self._is_anchor_grounded_in_citation(
                        source_anchor=slot_anchor,
                        citation=citation,
                        citation_span=text,
                    )
                    if not anchor_ok and slot_anchor in {"income statement", "balance sheet"}:
                        anchor_ok = any(
                            marker in lower
                            for marker in self._source_anchor_keywords(slot_anchor)
                        )
                    if (
                        not anchor_ok
                        and slot_anchor in {"income statement", "balance sheet"}
                        and metric_hit_count > 0
                        and (year_overlap_span or year_overlap_title or not slot_years)
                    ):
                        # Retrieval already applies entity/year constraints; allow
                        # statement-line rows that omit explicit anchor tokens.
                        anchor_ok = True
                if not anchor_ok:
                    continue

                candidate_values: list[tuple[float, int]] = []
                for term in metric_terms[:10]:
                    for m in re.finditer(re.escape(term), lower):
                        start = max(0, m.start() - 36)
                        end = min(len(text), m.end() + 140)
                        window = text[start:end]
                        if is_capex_cashflow_slot and any(
                            marker in window.lower() for marker in capex_ratio_markers
                        ):
                            continue
                        value = self._extract_primary_financial_number(window)
                        if value is None:
                            continue
                        candidate_values.append((float(value), 2))

                for year in slot_years:
                    for m in re.finditer(re.escape(year), text):
                        start = max(0, m.start() - 24)
                        end = min(len(text), m.end() + 110)
                        window = text[start:end]
                        if is_capex_cashflow_slot and any(
                            marker in window.lower() for marker in capex_ratio_markers
                        ):
                            continue
                        value = self._extract_primary_financial_number(window)
                        if value is None:
                            continue
                        candidate_values.append((float(value), 2))

                if not candidate_values:
                    value = self._extract_primary_financial_number(text)
                    if value is None:
                        continue
                    candidate_values.append((float(value), 0))

                metric_key = self._canonical_metric_key(slot_metric)
                for value, priority in candidate_values:
                    abs_value = abs(value)
                    if float(abs_value).is_integer() and 1900 <= abs_value <= 2100:
                        continue
                    if any(tok in metric_key for tok in ["revenue", "net sales", "net revenues", "total assets"]):
                        if abs_value < 100.0:
                            alt = self._extract_largest_non_year_number(text)
                            if alt is None or abs(alt) < 100.0:
                                continue
                            value = float(alt)
                            abs_value = abs(value)

                    score = priority + metric_hit_count * 2
                    if year_overlap_span:
                        score += 4
                    elif year_overlap_title:
                        score += 1
                    elif slot_years and not span_years and not title_years:
                        score += 1
                    if slot_anchor:
                        score += 1

                    if score > best_score:
                        value_text = self._format_numeric_for_slot_value(value)
                        if not value_text:
                            continue
                        if self._is_capex_metric(slot_metric) and slot_anchor == "cash flow statement":
                            value_text = self._positive_amount_string(value_text)
                        best_value = value_text
                        best_citation = citation
                        best_score = score

            if best_score < 0 or not best_value or not best_citation:
                continue
            entries.append(
                {
                    "slot": slot_raw,
                    "value": best_value,
                    "citation": best_citation,
                }
            )
        return entries

    def _answer_matches_calc_result(self, answer: str, calc_result: str) -> bool:
        answer_text = str(answer or "").strip()
        result_text = str(calc_result or "").strip()
        if not answer_text or not result_text:
            return False
        normalized_answer = answer_text.replace(",", "")
        normalized_result = result_text.replace(",", "")
        if normalized_result in normalized_answer:
            return True
        answer_num = extract_first_number(normalized_answer)
        result_num = extract_first_number(normalized_result)
        if answer_num is None or result_num is None:
            return False
        decimals = 0
        if "." in normalized_result:
            decimals = len(normalized_result.rsplit(".", 1)[1])
        tolerance = 10 ** (-(decimals + 1)) if decimals > 0 else 1e-6
        return abs(answer_num - result_num) <= tolerance

    def _build_calc_result_answer(
        self,
        query_state: dict[str, Any],
        evidence_ledger: list[dict[str, str]],
        context: str,
        calc_result: str,
    ) -> str:
        required_slots = self._required_slots(query_state)
        citations_in_context = self._context_citation_map(context)
        slot_to_citations: dict[str, list[str]] = {}
        for entry in evidence_ledger:
            slot_key = self._normalize_slot(entry.get("slot", ""))
            citation = str(entry.get("citation", "") or "").strip()
            if not slot_key or not citation:
                continue
            slot_to_citations.setdefault(slot_key, []).append(citation)

        selected: list[str] = []
        seen = set()
        for slot in required_slots:
            slot_key = self._normalize_slot(slot)
            candidates = slot_to_citations.get(slot_key, [])
            preferred = None
            for citation in candidates:
                if self._normalize_citation(citation) in citations_in_context:
                    preferred = citation
                    break
            if preferred is None and candidates:
                preferred = candidates[0]
            if preferred:
                key = self._normalize_citation(preferred)
                if key not in seen:
                    seen.add(key)
                    selected.append(preferred)

        if not selected:
            for entry in evidence_ledger:
                citation = str(entry.get("citation", "") or "").strip()
                if not citation:
                    continue
                key = self._normalize_citation(citation)
                if key in seen:
                    continue
                if citations_in_context and key not in citations_in_context:
                    continue
                seen.add(key)
                selected.append(citation)

        result_text = str(calc_result or "").strip()
        unit = str(query_state.get("unit", "") or "").strip().lower()
        if result_text and ("percent" in unit or unit in {"%", "pct", "percents"}):
            if not result_text.endswith("%"):
                result_text = f"{result_text}%"
        if selected:
            return "@@ANSWER: " + result_text + " " + " ".join(selected)
        return "@@ANSWER: " + result_text

    def _normalize_final_answer_for_query(
        self,
        answer: str,
        query_state: dict[str, Any],
    ) -> str:
        text = str(answer or "")
        if not text.strip() or self._is_insufficient_answer(text):
            return text
        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        if answer_type == "compute":
            return text

        top_metric_is_capex = self._is_capex_metric(query_state.get("metric", ""))
        slot_metric_is_capex = any(
            self._is_capex_metric(slot.get("metric", ""))
            for slot in (self._parse_slot_struct(slot_raw) for slot_raw in self._required_slots(query_state))
            if slot
        )
        if not (top_metric_is_capex or slot_metric_is_capex):
            return text

        anchors: list[str] = []
        top_anchor = str(query_state.get("source_anchor", "") or "").strip().lower()
        if top_anchor:
            anchors.append(top_anchor)
        for slot in self._required_slots(query_state):
            slot_struct = self._parse_slot_struct(slot)
            if not slot_struct:
                continue
            slot_anchor = str(slot_struct.get("source_anchor", "") or "").strip().lower()
            if slot_anchor:
                anchors.append(slot_anchor)
        if anchors and all(anchor != "cash flow statement" for anchor in anchors):
            return text

        prefix_match = re.search(r"@@answer:\s*", text, flags=re.IGNORECASE)
        if not prefix_match:
            return text
        head = text[:prefix_match.end()]
        body = text[prefix_match.end():]
        normalized_body = self._positive_amount_string(body)
        if normalized_body == body:
            return text
        return head + normalized_body


    @staticmethod
    def _validate_calc_plan_json(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "top-level must be JSON object"
        expression = str(data.get("expression", "") or "").strip()
        if not expression:
            return False, "expression must be non-empty string"
        precision = data.get("precision", None)
        if precision is not None:
            try:
                int(precision)
            except Exception:
                return False, "precision must be int or null"
        return True, ""

    @staticmethod
    def _calc_plan_retry_message(failed_output: Any, reason: str) -> str:
        return CALCULATION_PLAN_RETRY_PROMPT.format(
            error=reason,
            previous_output=compact_json(failed_output, max_chars=900),
        )

    async def _compute_with_calculator_from_ledger(self, state: AgentState) -> Optional[dict[str, Any]]:
        if str(state.query_state.get("answer_type", "")).lower() != "compute":
            return None
        if state.missing_slots:
            return None
        if not state.evidence_ledger:
            return None

        prompt = CALCULATION_PLAN_PROMPT.format(
            query=state.user_query,
            query_state=self._compact_json(state.query_state, max_chars=1200),
            evidence_ledger=self._compact_json({"entries": state.evidence_ledger}, max_chars=2400),
        )
        messages = [{"role": "user", "content": prompt}]
        data, ok, _ = await generate_json_with_retries(
            self.llm,
            messages,
            self._validate_calc_plan_json,
            self._calc_plan_retry_message,
            max_attempts=3,
            logger=logger,
            warning_prefix="calculation plan json generation failed",
            model=self.stage_model,
        )
        if not ok:
            return None

        expression = str(data.get("expression", "") or "").strip()
        required_slots = self._required_slots(state.query_state)
        min_ground_matches = 2 if len(required_slots) >= 3 else 1
        grounded_number_matches = self._expression_ledger_match_count(
            expression,
            state.evidence_ledger,
        )
        if grounded_number_matches < min_ground_matches:
            return {
                "ok": False,
                "error": "expression_not_grounded_in_ledger",
                "expression": expression,
                "required_slot_count": len(required_slots),
                "matched_ledger_values": grounded_number_matches,
            }

        precision = data.get("precision", None)
        if precision is None and state.query_state.get("rounding") is not None:
            precision = state.query_state.get("rounding")
        calc = self._call_calculator(expression, precision=precision)
        if not calc.get("ok"):
            return calc

        calc_text = (
            f"Calculator expression: {calc.get('expression')}\n"
            f"Calculator result: {calc.get('result')}"
        )
        calc_chunk = {
            "title": "CALCULATOR",
            "page": 0,
            "sent_id": -200000 - len(state.evidence_ledger),
            "text": calc_text,
        }
        state.all_context_data.append(calc_chunk)
        state.history.append({
            "role": "tool",
            "tool_call_id": "compute_auto_calc",
            "name": "calculator",
            "content": calc_text,
        })
        return calc
