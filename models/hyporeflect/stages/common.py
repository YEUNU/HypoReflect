import re
from typing import Any, Optional


_ANSWER_PREFIX_RE = re.compile(r"@@ANSWER\s*:", flags=re.IGNORECASE)
CITATION_RE = re.compile(
    r"\[\[[^\]]+,\s*(?:Page\s*\d+\s*,\s*Chunk\s*\d+|\d+)\s*\]\]",
    flags=re.IGNORECASE,
)
NUMERIC_QUERY_MARKERS = (
    "what is",
    "how much",
    "round",
    "calculate",
    "average",
    "year-over-year",
    "yoy",
    "ratio",
    "%",
    "percent",
    "margin",
    "turnover",
    "dpo",
    "roa",
)
NUMERIC_METRIC_KEYS = {
    "quick ratio",
    "fixed asset turnover",
    "return on assets",
    "days payable outstanding",
    "dividend payout ratio",
    "operating margin",
    "gross margin",
    "capital expenditure",
    "revenue",
    "cogs",
}
_VALID_MISSING_DATA_POLICIES = {
    "insufficient",
    "zero_if_not_explicit",
    "inapplicable_explain",
}


def extract_final_answer_from_json(data: Any) -> tuple[Optional[str], str]:
    if not isinstance(data, dict):
        return None, "top-level must be object"
    text = data.get("final_answer", None)
    if not isinstance(text, str):
        return None, "missing string field 'final_answer'"
    answer = text.strip()
    if not answer:
        return None, "empty final_answer"
    if len(_ANSWER_PREFIX_RE.findall(answer)) > 1:
        return None, "multiple @@ANSWER prefixes are not allowed"
    if "@@ANSWER:" not in answer:
        answer = f"@@ANSWER: {answer}"
    return answer, ""


def normalize_missing_data_policy(value: Any) -> str:
    policy = str(value or "insufficient").strip().lower()
    if policy in _VALID_MISSING_DATA_POLICIES:
        return policy
    return "insufficient"


def missing_data_policy(state: Any) -> str:
    if isinstance(state, dict):
        query_state = state
    else:
        query_state = getattr(state, "query_state", None)
    if not isinstance(query_state, dict):
        query_state = {}
    return normalize_missing_data_policy(query_state.get("missing_data_policy"))


def extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except Exception:
        return None


def format_retrieved_chunks(
    nodes: Any,
    *,
    max_chunks: int = 12,
    per_chunk_chars: int = 380,
    total_chars: int = 4500,
) -> str:
    """Render the full retrieval pool for reflection/refinement prompts.

    Dedupes by (title, page, sent_id), caps each chunk's text, then enforces
    a total character budget. Empty / non-list input renders as the literal
    string `"none"` so the prompt placeholder is never blank.
    """
    if not isinstance(nodes, list) or not nodes:
        return "none"
    seen: set[tuple[str, Any, Any]] = set()
    lines: list[str] = []
    total = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        title = str(node.get("title") or node.get("doc") or "Unknown").strip()
        page = node.get("page", 0)
        sent_id = node.get("sent_id", 0)
        key = (title, page, sent_id)
        if key in seen:
            continue
        seen.add(key)
        text = re.sub(r"\s+", " ", str(node.get("text", "") or "")).strip()
        if not text:
            continue
        snippet = text[:per_chunk_chars]
        if len(text) > per_chunk_chars:
            snippet = snippet.rstrip() + "…"
        line = f"[[{title}, Page {page}, Chunk {sent_id}]] {snippet}"
        if total + len(line) + 1 > total_chars and lines:
            break
        lines.append(line)
        total += len(line) + 1
        if len(lines) >= max_chunks:
            break
    return "\n".join(lines) if lines else "none"


def answer_matches_calc_result(answer: str, calc_result: str) -> bool:
    """Check whether the calculator result is faithfully reflected in the
    answer text. Used by both the reflection arithmetic-check pass and the
    execution synthesis "calculator-direct" gate to keep their decisions
    consistent. Tolerance scales with the calc result's decimal precision
    (10^-(decimals+1), or 1e-6 for integer results).
    """
    answer_text = str(answer or "").strip().replace(",", "")
    result_text = str(calc_result or "").strip().replace(",", "")
    if not answer_text or not result_text:
        return False
    if result_text in answer_text:
        return True
    answer_num = extract_first_number(answer_text)
    result_num = extract_first_number(result_text)
    if answer_num is None or result_num is None:
        return False
    decimals = 0
    if "." in result_text:
        decimals = len(result_text.rsplit(".", 1)[1])
    tolerance = 10 ** (-(decimals + 1)) if decimals > 0 else 1e-6
    return abs(answer_num - result_num) <= tolerance
