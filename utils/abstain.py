"""Shared abstain-phrase detection for FinanceBench-style refusal scoring.

FinanceBench (Islam et al., arXiv:2311.11944) uses a human-annotated 3-way
taxonomy `Correct Answer | Incorrect Answer | Refusal` (visible in the
`label` field of the official result JSONLs at
https://github.com/patronus-ai/financebench/tree/main/results).

The local LLM-as-judge approximation distinguishes the same three categories
via `(llm_judge_score, hallucination, answer_attempted)`. Refusal detection
previously matched only the literal substring "insufficient evidence", which
is Hypo's pipeline-specific abstain phrase. HopRAG and naive baselines use
natural-language abstains ("I do not know", "the context does not contain
...") and were therefore mis-classified as `Incorrect Answer` (with inflated
`answer_attempted = 1`).

This module centralizes the abstain phrase list so `utils/metrics.py`,
`tools/benchmark_report.py`, and `cli/benchmark.py` apply the same rule.
"""
from __future__ import annotations

# Lowercased substring patterns. Substring match — no regex, no token-level
# parsing. The list intentionally covers (a) Hypo's prefixed abstain phrase,
# (b) HopRAG's natural-language abstains, and (c) common phrasings that all
# three RAG strategies' synthesis prompts converge on when CONTEXT lacks the
# queried fact.
ABSTAIN_PHRASES: tuple[str, ...] = (
    "insufficient evidence",
    "i do not know",
    "i don't know",
    "do not know",
    "cannot be determined",
    "cannot determine",
    "unable to determine",
    "not determinable",
    "not provided in the context",
    "not specified in the context",
    "not stated in the context",
    "not mentioned in the context",
    "no information",
    "the context does not contain",
    "the context does not mention",
    "the context does not include",
    "the context does not provide",
    "context provides no",
    "context lacks",
    "unable to find relevant information",
    "unable to find",
    "no relevant information",
)


def is_abstain(text) -> bool:
    """True if `text` contains any recognized abstain phrase.

    Matches lowercased substring; safe to call on the FULL response or on
    an extracted-final-answer slice. CoT responses that include an abstain
    token mid-reasoning but conclude with a substantive answer should be
    checked against the extracted final answer (see
    `cli/benchmark.py::_extract_final_answer`), not the full text.
    """
    return any(p in str(text or "").lower() for p in ABSTAIN_PHRASES)


def financebench_label(judge_score, response: str) -> str:
    """Map a (judge_score, response) pair to the FinanceBench 3-way label.

    Judge score takes precedence over abstain detection: when ground truth
    is itself "no value / not disclosed", an abstention IS the correct
    answer and the judge prompt awards score=1.0 for it (see judge prompt
    rule 3 in `utils/prompts/evaluation.py`). Downgrading such answers to
    `Refusal` would mis-classify FinanceBench's official `Correct Answer`
    label as `Refusal`.

    - "Correct Answer":  judge_score >= 0.5 (irrespective of abstain
      phrase, since GT may itself be a non-answer).
    - "Refusal":         judge_score < 0.5 AND response contains an
      abstain phrase (honest "I do not know" against a substantive GT).
    - "Incorrect Answer": judge_score < 0.5 AND substantive wrong answer.
    """
    try:
        score = float(judge_score)
    except Exception:
        score = 0.0
    if score >= 0.5:
        return "Correct Answer"
    if is_abstain(response):
        return "Refusal"
    return "Incorrect Answer"
