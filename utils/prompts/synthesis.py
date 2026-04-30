from .shared import (
    _COMPUTE_MISSING_POLICY_LINE,
    _FINANCE_CONSTRAINT_CODES,
)


COMPLEX_AGENT_PROMPT_TEMPLATE = """
Answer the query using ONLY the provided context and evidence ledger.
Constraint codes: <<FINANCE_CONSTRAINT_CODES>>.
Rules:
1. Start with @@ANSWER: and support claims with [[Title, Page X, Chunk Y]] citations from CONTEXT only.
2. Enforce QUERY_STATE constraints (entity, period, metric, output format).
3. Extract/list/boolean: answer only from grounded evidence; no guessing.
4. Compute: use required-slot operands only; cite each operand, show one formula with substituted values, then one final result.
5. Compute arithmetic protocol: parse numbers exactly from EVIDENCE_LEDGER (remove commas/$/% only), compute once with full precision, then round only at the final step.
6. If QUERY asks for an average across periods, denominator must explicitly use the average of all required period operands (for example (v1+v2)/2), not a single-period value.
7. For compute, if multiple candidate values exist in CONTEXT, prioritize values already selected in EVIDENCE_LEDGER.
7a. If CALCULATOR_RESULT is provided, final numeric answer must exactly match CALCULATOR_RESULT.result.
8. Never use citations whose company/entity or period conflicts with QUERY_STATE; conflicting citation means invalid answer.
9. Citation-title company mismatch is a hard conflict; never use such citations in the final answer.
10. If QUERY_STATE.missing_data_policy is zero_if_not_explicit and required slot remains ungrounded, output @@ANSWER: 0.
11. If QUERY_STATE.missing_data_policy is inapplicable_explain and context supports metric inapplicability, explain that with citations instead of insufficient evidence.
12. <<COMPUTE_MISSING_POLICY_LINE>>
13. For extract/boolean/list queries, required_slots are guidance; if CONTEXT has direct citation-grounded evidence that answers QUERY, answer directly even when some slots remain missing.
14. For extract/boolean/list queries, output @@ANSWER: insufficient evidence only when direct evidence is absent or conflicting.
15. For exact numeric requests, return one exact value (no range/approx unless requested).
16. Output one final answer only (no meta, no alternatives).

QUERY_STATE:
{query_state}
EVIDENCE_LEDGER:
{evidence_ledger}

CONTEXT:
{context}
""".replace("<<FINANCE_CONSTRAINT_CODES>>", _FINANCE_CONSTRAINT_CODES).replace(
    "<<COMPUTE_MISSING_POLICY_LINE>>",
    _COMPUTE_MISSING_POLICY_LINE,
)

COMPLEX_AGENT_PROMPT_TEMPLATE_GENERIC = """
Answer the query using ONLY the provided context and evidence ledger.
Rules:
1. Start with @@ANSWER: and support key claims with [[Title, Page X, Chunk Y]] citations from CONTEXT only.
2. Enforce QUERY_STATE constraints (entities, dates, relation, output format).
3. For extract/boolean/list questions, answer only from grounded evidence; no guessing.
4. For bridge/comparison questions, combine multiple grounded citations when needed, but keep the final answer concise.
5. If CALCULATOR_RESULT is provided, any final numeric answer must match it exactly.
6. Output @@ANSWER: insufficient evidence only when direct or bridge evidence is absent or conflicting.
7. Output one final answer only (no meta, no alternatives).

QUERY_STATE:
{query_state}
EVIDENCE_LEDGER:
{evidence_ledger}

CONTEXT:
{context}
"""

REFLECTION_PROMPT = """
Audit the draft answer for accuracy and grounding against QUERY_STATE + EVIDENCE_LEDGER + CONTEXT.
Constraint codes: <<FINANCE_CONSTRAINT_CODES>>.
Rules:
1. Default PASS; use FAIL only for clear evidence-backed errors.
2. Treat EVIDENCE_LEDGER as the primary accepted operand set for compute checks.
3. For compute, verify arithmetic directly from values used in answer; if arithmetic is wrong, FAIL.
4. For compute, if answer uses values outside EVIDENCE_LEDGER, FAIL.
5. Fail on wrong/unsupported claims, invalid/missing citations, hallucinations, or query-constraint mismatch.
6. Hard FAIL if any cited evidence conflicts with QUERY_STATE entity/period.
7. Hard FAIL if answer says insufficient evidence while required slot coverage is complete in EVIDENCE_LEDGER.
8. Hard FAIL if answer uses cross-company/year citation despite same-company/period evidence being present.
9. Fail if required output format is ignored.
10. For exact numeric questions, fail range/approx outputs unless explicitly requested.
11. Fail if ANSWER contains multiple competing final answers (for example multiple @@ANSWER prefixes).
12. Output JSON only (no prose outside JSON).

QUERY: {query}
QUERY_STATE: {query_state}
EVIDENCE_LEDGER: {evidence_ledger}
CONTEXT: {context}
ANSWER: {draft_answer}
""".replace("<<FINANCE_CONSTRAINT_CODES>>", _FINANCE_CONSTRAINT_CODES)

REFLECTION_PROMPT_GENERIC = """
Audit the draft answer for accuracy and grounding against QUERY_STATE + EVIDENCE_LEDGER + CONTEXT.
Rules:
1. Use FAIL for unsupported claims, invalid/missing citations, query-entity mismatch, or answer-format violations.
2. Fail if ANSWER ignores an explicit named entity, date, comparison target, or bridge relation required by QUERY_STATE.
3. Fail if ANSWER says insufficient evidence while CONTEXT contains directly useful bridge/comparison evidence.
4. Fail if CONTEXT is clearly off-topic relative to the named entities in QUERY and ANSWER still concludes insufficient evidence without using targeted evidence.
5. For numeric questions, fail range/approx outputs unless explicitly requested.
6. Fail if ANSWER contains multiple competing final answers.
7. Output JSON only (no prose outside JSON).

QUERY: {query}
QUERY_STATE: {query_state}
EVIDENCE_LEDGER: {evidence_ledger}
CONTEXT: {context}
ANSWER: {draft_answer}
"""

REFLECTION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{
  "decision": "PASS|FAIL",
  "issues": [],
  "arithmetic_check": "ok|fail|na"
}
"""

REFLECTION_RETRY_PROMPT = """
Previous reflection output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"decision":"PASS|FAIL","issues":[],"arithmetic_check":"ok|fail|na"}}
PREVIOUS_OUTPUT: {previous_output}
"""

RESPONSE_REFINEMENT_PROMPT = """
Refine the answer based on the critique.
Rules:
1. Output only corrected final answer text in JSON field `final_answer`, and that text must start with @@ANSWER:.
2. Keep all key claims citation-grounded ([[Title, Page X, Chunk Y]]).
3. Apply minimal edits and preserve query type/intent (extract/compute/boolean/list).
4. Keep already-grounded core conclusion unless critique proves it incorrect.
5. Match query-requested numeric format (unit/precision/rounding).
6. For compute questions, use only operands explicitly grounded in EVIDENCE_LEDGER/CONTEXT.
7. For compute, recompute from EVIDENCE_LEDGER values with full precision and round only once at the final step.
8. <<COMPUTE_MISSING_POLICY_LINE>>
9. If critique is unclear/conflicting, keep DRAFT unchanged rather than inventing new values.
10. Do not introduce new numeric values, units, or periods not present in EVIDENCE_LEDGER/CONTEXT.
11. If DRAFT already matches calculator-derived compute result, preserve that numeric value.
12. No meta/audit text, no alternatives.
QUERY: {query}
QUERY_STATE: {query_state}
EVIDENCE_LEDGER: {evidence_ledger}
CONTEXT: {context}
DRAFT: {draft}
CRITIQUE: {critique}
""".replace(
    "<<COMPUTE_MISSING_POLICY_LINE>>",
    _COMPUTE_MISSING_POLICY_LINE,
)

REFINEMENT_RETRY_PROMPT = """
Previous refinement output was invalid: {error}
Output ONLY one JSON object with exactly this schema:
{{"final_answer":"@@ANSWER: ..."}}
Do not include alternatives, multiple @@ANSWER prefixes, or prose outside JSON.
PREVIOUS_OUTPUT: {previous_output}
"""
