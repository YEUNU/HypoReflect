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
13a. Intent preservation: when the literal compute is blocked because one required operand is missing, BUT CONTEXT contains grounded narrative that addresses the QUERY's underlying intent (drivers, causes, qualitative direction, comparative ranking), answer the underlying intent with citations rather than abstaining. Concretely: a "what drove X change" query should report the cause clauses present in CONTEXT even when the change-magnitude operand is missing; a "is metric Y healthy" query should report the qualitative read present in CONTEXT even when one component of the ratio is missing. Only emit `insufficient evidence` when neither the literal computation nor the underlying intent has any grounded support in CONTEXT.
13b. No concept substitution: do NOT answer an adjacent question. If QUERY asks segment-by-revenue and CONTEXT only has cash-flow-by-activity, the answer is `insufficient evidence`, not the cash-flow ranking. Intent preservation (13a) only applies when CONTEXT directly addresses the queried subject.
14. For extract/boolean/list queries, output @@ANSWER: insufficient evidence only when direct evidence is absent or conflicting AND the underlying intent (per 13a) is also ungrounded.
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

REFLECTION_PROMPT = """
Audit the draft answer for accuracy and grounding against QUERY_STATE + EVIDENCE_LEDGER + CONTEXT.
Constraint codes: <<FINANCE_CONSTRAINT_CODES>>.

Process: BEFORE assigning a verdict, internally run the four checks below
and record what each produced under `checks_performed` (free-text findings,
one entry per check). Default to PASS only when every check produced no
counter-evidence; if any check surfaces a concrete defect cited from
QUERY_STATE / EVIDENCE_LEDGER / CONTEXT, the verdict is FAIL.

Checks:
(A) Arithmetic & operand identity
    - For compute, recompute from EVIDENCE_LEDGER values; arithmetic must match.
    - Operand-slot provenance: each numeric operand must come from the
      EVIDENCE_LEDGER entry whose `slot` tag matches both the operand's
      metric and period. Wrong-slot reuse (e.g., using one period's value
      for another period's slot) → `operand_slot_mismatch`.
    - Operand-magnitude sanity: when the queried metric is a totals/aggregate
      concept and the chosen operand is conspicuously smaller than another
      same-metric same-period candidate present in EVIDENCE_LEDGER / CONTEXT
      primary statement, prefer the primary total → `operand_magnitude_anomaly`.
    - Formula-identity: when QUERY names a specific ratio/margin/coverage
      (anything ending in "ratio", "margin", "yield", "turnover", "coverage"),
      the operand set actually used in ANSWER must be consistent with the
      standard definition of that named metric. If the operand set matches
      a different (similarly-named) metric's definition instead, FAIL with
      `formula_identity_mismatch`.
(B) Enumeration coverage vs CONTEXT/EVIDENCE_LEDGER
    - For extract/list/"what drove X" style queries: if EVIDENCE_LEDGER or
      CONTEXT contains multiple grounded items the answer should report
      (in the same sentence/list/table), the answer must include all such
      grounded items. Omitting any → `incomplete_enumeration`. Do NOT fail
      when the missing items are absent from EVIDENCE_LEDGER/CONTEXT.
    - Restatement check: for "what drove"/"what caused" queries, an answer
      that merely restates the metric or its delta without citing any
      cause/driver clause present in CONTEXT → `restated_metric_no_drivers`.
(C) Intent alignment
    - Does the ANSWER address the QUERY's underlying intent, or does it
      substitute an adjacent concept (e.g., reporting cash-flow-by-activity
      when QUERY asks segment-by-revenue)? Concept substitution → FAIL.
    - "Insufficient evidence" with the underlying intent grounded in
      CONTEXT (drivers, qualitative direction) is also a FAIL — the answer
      should report the intent-level finding rather than abstain.
(D) Citation, entity/period, format
    - Hard FAIL on cross-company / cross-period citations when same-company
      / same-period evidence is present.
    - Hard FAIL on missing/invalid inline citations, multiple competing
      @@ANSWER prefixes, or required output-format violations.
    - For exact numeric questions, range/approximate outputs FAIL unless
      explicitly requested.
    - "Insufficient evidence" with required slot coverage complete in
      EVIDENCE_LEDGER → FAIL.

Output JSON only (no prose outside JSON).

QUERY: {query}
QUERY_STATE: {query_state}
EVIDENCE_LEDGER: {evidence_ledger}
CONTEXT: {context}
ANSWER: {draft_answer}
""".replace("<<FINANCE_CONSTRAINT_CODES>>", _FINANCE_CONSTRAINT_CODES)

REFLECTION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{
  "checks_performed": ["A: ...", "B: ...", "C: ...", "D: ..."],
  "decision": "PASS|FAIL",
  "issues": [],
  "arithmetic_check": "ok|fail|na"
}
"""

REFLECTION_RETRY_PROMPT = """
Previous reflection output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"checks_performed":["A: ...","B: ...","C: ...","D: ..."],"decision":"PASS|FAIL","issues":[],"arithmetic_check":"ok|fail|na"}}
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
