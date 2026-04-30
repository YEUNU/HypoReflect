import logging
import re
import string
from difflib import SequenceMatcher
from typing import List, Any, Optional
from core.config import RAGConfig
from utils.prompts import FINANCEBENCH_HALLUCINATION_PROMPT, FINANCEBENCH_JUDGE_PROMPT

logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Normalize answer text for comparison."""
    if not s: return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# --- FinanceBench Specific Metrics ---

def extract_numeric_value(s: str) -> float | None:
    """
    금융 값에서 숫자 추출.
    "$1,577.00" → 1577.0
    "8.70 billion" → 8.7 (단위 변환은 별도 처리 필요)
    """
    if not s:
        return None
    
    # 통화 기호 및 쉼표 제거
    cleaned = re.sub(r'[$€£¥,]', '', s.strip())
    
    # 숫자 패턴 매칭 (음수 포함)
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def calculate_financebench_accuracy(prediction: str, ground_truth: str) -> dict:
    """
    FinanceBench 금융 값 정확도 계산.
    
    Returns:
        dict with 'exact_match', 'numeric_match', 'contains_match'
    """
    if not prediction or not ground_truth:
        return {"exact_match": 0.0, "numeric_match": 0.0, "contains_match": 0.0}
    
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    # 1. Exact Match (정규화 후)
    exact_match = 1.0 if pred_norm == gt_norm else 0.0
    
    # 2. Numeric Match (숫자 추출 후 비교)
    pred_num = extract_numeric_value(prediction)
    gt_num = extract_numeric_value(ground_truth)
    
    numeric_match = 0.0
    if pred_num is not None and gt_num is not None:
        # 상대 오차 5% 이내면 매칭
        if gt_num != 0:
            rel_error = abs(pred_num - gt_num) / abs(gt_num)
            numeric_match = 1.0 if rel_error < 0.05 else 0.0
        else:
            numeric_match = 1.0 if pred_num == 0 else 0.0
    
    # 3. Contains Match (ground truth가 prediction에 포함)
    contains_match = 1.0 if gt_norm in pred_norm else 0.0
    
    return {
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "contains_match": contains_match
    }


def calculate_evidence_match(
    retrieved_sources: List[Any], 
    expected_doc: str, 
    expected_page: int | None = None
) -> dict:
    """
    FinanceBench 증거 매칭 - 문서/페이지 레벨.
    Supports both string filenames and structured [title, page, ...] lists.
    
    Args:
        retrieved_sources: List of strings or lists [title, page, sent_id]
        expected_doc: 예상 문서명 (e.g., "3M_2018_10K")
        expected_page: 예상 페이지 번호 (optional)
    
    Returns:
        dict with 'doc_match', 'page_match'
    """
    if not retrieved_sources or not expected_doc:
        return {"doc_match": 0.0, "page_match": 0.0}
    
    doc_match = 0.0
    page_match = 0.0

    def normalize_doc_id(value: str) -> str:
        if not value:
            return ""
        lowered = str(value).lower().strip()
        lowered = re.sub(r"\.(pdf|txt|md|json)$", "", lowered)
        lowered = lowered.replace("10-k", "10k").replace("10-q", "10q")
        lowered = re.sub(r"[^a-z0-9]+", "", lowered)
        return lowered

    def tokenize_doc_id(value: str) -> set[str]:
        if not value:
            return set()
        lowered = str(value).lower()
        lowered = lowered.replace("10-k", "10k").replace("10-q", "10q")
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return {tok for tok in lowered.split() if tok}

    expected_doc_norm = normalize_doc_id(expected_doc)
    expected_doc_tokens = tokenize_doc_id(expected_doc)

    for source in retrieved_sources:
        src_title = ""
        src_page = None

        # Dict Source: {"doc": ..., "page": ..., "text": ...}
        if isinstance(source, dict):
            src_title = str(source.get("doc", "")).lower()
            src_page = source.get("page")
        
        # Structured Source: [title, page, sent_id]
        elif isinstance(source, (list, tuple)) and len(source) >= 2:
            src_title = str(source[0]).lower()
            src_page = source[1]
            
        # String Source: "Title" or "Title_page_5"
        elif isinstance(source, str):
            src_title = source

        src_doc_norm = normalize_doc_id(src_title)
        src_doc_tokens = tokenize_doc_id(src_title)

        is_doc_match = False
        if expected_doc_norm and src_doc_norm:
            if expected_doc_norm in src_doc_norm or src_doc_norm in expected_doc_norm:
                is_doc_match = True
            else:
                sim = SequenceMatcher(None, expected_doc_norm, src_doc_norm).ratio()
                if sim >= 0.92:
                    is_doc_match = True

        if not is_doc_match and expected_doc_tokens and src_doc_tokens:
            overlap = len(expected_doc_tokens.intersection(src_doc_tokens))
            min_required = max(1, int(len(expected_doc_tokens) * 0.6))
            if overlap >= min_required:
                is_doc_match = True

        if is_doc_match:
            doc_match = 1.0
            if expected_page is not None:
                if isinstance(src_page, (int, float)) and int(src_page) == expected_page:
                    page_match = 1.0
                    break
                if isinstance(source, str):
                    source_lower = source.lower()
                    page_pattern = f"page_{expected_page:03d}" if isinstance(expected_page, int) else f"page_{expected_page}"
                    if page_pattern in source_lower or f"_page_{expected_page}" in source_lower:
                        page_match = 1.0
                        break

    return {"doc_match": doc_match, "page_match": page_match}

async def evaluate_financebench_response(
    query: str,
    response: str,
    ground_truth: str,
    retrieved_sources: List[Any],
    expected_doc: str,
    expected_page: Optional[int] = None,
    vllm_client = None
) -> dict:
    """
    FinanceBench 통합 평가 인터페이스 (LLM-as-a-judge + Evidence Match).
    """
    judge_score = 0.0
    judge_reason = ""
    hallucination = 0.0
    hallucination_reason = ""
    hallucination_source = ""
    hallucination_model = str(RAGConfig.HALLUCINATION_EVAL_MODEL or "").strip() or RAGConfig.EVAL_MODEL

    def _parse_judge_score(raw_score: Any) -> Optional[float]:
        try:
            if raw_score is None:
                return None
            score = float(raw_score)
            return max(0.0, min(1.0, score))
        except Exception:
            return None

    def _parse_hallucination_score(raw_score: Any, raw_label: Any = None) -> Optional[float]:
        parsed = _parse_judge_score(raw_score)
        if parsed is not None:
            return 1.0 if parsed >= 0.5 else 0.0
        label = str(raw_label or "").strip().lower()
        if label in {"hallucinated", "hallucination", "yes", "true", "1"}:
            return 1.0
        if label in {"not_hallucinated", "not hallucinated", "no", "false", "0"}:
            return 0.0
        return None

    def _is_insufficient_text(text: Any) -> bool:
        return "insufficient evidence" in str(text or "").lower()

    if vllm_client:
        judge_prompt = FINANCEBENCH_JUDGE_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            response=response
        )
        try:
            # Primary judge model
            res_json = await vllm_client.generate_json(
                [{"role": "user", "content": judge_prompt}],
                model=RAGConfig.EVAL_MODEL
            )
            parsed = _parse_judge_score(res_json.get("score"))

            # Fallback to local generation model when judge score is missing/invalid.
            if parsed is None:
                fallback_model = RAGConfig.DEFAULT_MODEL
                if fallback_model and fallback_model != RAGConfig.EVAL_MODEL:
                    logger.warning(
                        "Judge response missing score with model '%s'. Retrying with fallback model '%s'.",
                        RAGConfig.EVAL_MODEL,
                        fallback_model,
                    )
                    res_json = await vllm_client.generate_json(
                        [{"role": "user", "content": judge_prompt}],
                        model=fallback_model
                    )
                    parsed = _parse_judge_score(res_json.get("score"))

            if parsed is not None:
                judge_score = parsed
                judge_reason = str(res_json.get("reason", ""))
            else:
                # Deterministic fallback instead of hard-zero when judge model is unavailable.
                fallback_acc = calculate_financebench_accuracy(response, ground_truth)
                judge_score = max(
                    fallback_acc["exact_match"],
                    fallback_acc["numeric_match"],
                    fallback_acc["contains_match"],
                )
                judge_reason = (
                    "fallback_heuristic: exact/numeric/contains max used due unavailable judge score"
                )
                logger.warning(
                    "LLM judge score unavailable from both primary='%s' and fallback='%s'. "
                    "Using heuristic score=%.2f",
                    RAGConfig.EVAL_MODEL,
                    RAGConfig.DEFAULT_MODEL,
                    judge_score,
                )
        except Exception as e:
            logger.error(f"LLM Judge failed: {e}")
            fallback_acc = calculate_financebench_accuracy(response, ground_truth)
            judge_score = max(
                fallback_acc["exact_match"],
                fallback_acc["numeric_match"],
                fallback_acc["contains_match"],
            )
            judge_reason = "fallback_heuristic_after_exception"
    else:
        fallback_acc = calculate_financebench_accuracy(response, ground_truth)
        judge_score = max(
            fallback_acc["exact_match"],
            fallback_acc["numeric_match"],
            fallback_acc["contains_match"],
        )
        judge_reason = "fallback_heuristic_without_judge_client"

    # Hallucination evaluation (GPT-5.2 by default): compare model response vs ground truth.
    if _is_insufficient_text(response):
        hallucination = 0.0
        hallucination_reason = "non_answer_insufficient"
        hallucination_source = "rule_non_answer"
    elif str(response or "").strip():
        if vllm_client:
            hallucination_prompt = FINANCEBENCH_HALLUCINATION_PROMPT.format(
                query=query,
                ground_truth=ground_truth,
                response=response,
            )
            try:
                hallu_json = await vllm_client.generate_eval_json(
                    [{"role": "user", "content": hallucination_prompt}],
                    model=hallucination_model,
                )
                parsed_hallucination = _parse_hallucination_score(
                    hallu_json.get("hallucination"),
                    hallu_json.get("label"),
                )
                if parsed_hallucination is not None:
                    hallucination = parsed_hallucination
                    hallucination_reason = str(hallu_json.get("reason", ""))
                    hallucination_source = "gpt_hallucination_judge"
                else:
                    hallucination = 1.0 if judge_score < 1.0 else 0.0
                    hallucination_reason = "fallback_llm_judge_due_invalid_hallucination_payload"
                    hallucination_source = "llm_judge_fallback"
            except Exception as e:
                logger.warning(
                    "Hallucination judge failed with model '%s': %s. Falling back to llm_judge_score.",
                    hallucination_model,
                    e,
                )
                hallucination = 1.0 if judge_score < 1.0 else 0.0
                hallucination_reason = "fallback_llm_judge_after_hallucination_exception"
                hallucination_source = "llm_judge_fallback"
        else:
            hallucination = 1.0 if judge_score < 1.0 else 0.0
            hallucination_reason = "fallback_llm_judge_without_judge_client"
            hallucination_source = "llm_judge_fallback"
    else:
        hallucination = 0.0
        hallucination_reason = "non_answer_empty"
        hallucination_source = "rule_non_answer"

    # Calculate Evidence Match (Doc & Page)
    # Supports dict/list/str source types directly.
    evidence_metrics = calculate_evidence_match(retrieved_sources, expected_doc, expected_page)

    return {
        "llm_judge_score": judge_score,
        "llm_judge_reason": judge_reason,
        "hallucination": hallucination,
        "hallucination_reason": hallucination_reason,
        "hallucination_source": hallucination_source,
        "hallucination_model": hallucination_model,
        "doc_match": evidence_metrics["doc_match"],
        "page_match": evidence_metrics["page_match"]
    }
