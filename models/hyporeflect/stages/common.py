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
