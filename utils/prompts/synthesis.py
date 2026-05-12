from .shared import (
    _COMPUTE_MISSING_POLICY_LINE,
    _FINANCE_CONSTRAINT_CODES,
)


# Synthesis prompt. CONTEXT is the primary source of truth; EVIDENCE_LEDGER is
# an advisory hint for compute-operand selection. The prior version (16 rules
# + 13a/13b/13c sub-rules) over-gated honest answers, so it has been collapsed
# to one principle-per-rule. Keep this prompt short — the local 4B model
# follows short rules and ignores long ones.
COMPLEX_AGENT_PROMPT_TEMPLATE = """
Answer the QUERY using CONTEXT. EVIDENCE_LEDGER is an advisory hint for
compute operands; it is NOT a gate. If a candidate value for the queried
fact is present in CONTEXT, answer with it even when EVIDENCE_LEDGER is
empty or missing that slot.

Rules:
1. Start the answer with `@@ANSWER:` and cite every factual or numeric claim
   inline as `[[Title, Page X, Chunk Y]]` using IDs printed in CONTEXT.
2. Honor QUERY_STATE entity, period, metric, and output format. Never cite
   a chunk whose company/period conflicts with QUERY_STATE.
3. Standard line-item synonyms refer to the same number — answer using the
   value present in CONTEXT regardless of which synonym the question uses:
     capex ≡ purchases of (additions to) property, plant and equipment
            from the investing-activities section of the cash flow statement.
     revenue ≡ net sales ≡ net revenues (top-line income statement).
     cost of revenue ≡ cost of sales ≡ cost of goods sold.
     net income ≡ net earnings (bottom-line income statement).
     operating income ≡ operating profit ≡ income from operations.
     gross profit ≡ gross margin (dollars).
   Do NOT bridge unrelated metrics (e.g., operating cash flow ≠ net income).
4. Compute: parse numbers verbatim from CONTEXT (strip commas/$/%), compute
   once at full precision, round only at the final step, and show one
   formula with substituted values then the final result.
5. Compute averages across periods must use the average of all required
   period operands, e.g. (v1+v2)/2, not a single-period value.
6. If CALCULATOR_RESULT is provided, the final numeric value must equal
   CALCULATOR_RESULT.result exactly.
7. {compute_missing_policy_line}
8. {_extraction_zero_policy}
9. For extract/boolean/list, answer directly from CONTEXT citations. Only
   emit `@@ANSWER: insufficient evidence` when no chunk in CONTEXT contains
   a candidate for the queried fact AND no underlying intent (drivers,
   qualitative direction) is grounded in CONTEXT.
10. No concept substitution: do not answer an adjacent question. If QUERY
    asks segment-by-revenue and CONTEXT only contains cash-flow-by-activity,
    abstain rather than reporting the cash-flow ranking.
11. For exact numeric requests, return one exact value (no range/approx).
12. Output one final answer only (no meta, no alternatives).

Constraint codes: {_finance_constraint_codes}.

QUERY_STATE:
{{query_state}}
EVIDENCE_LEDGER:
{{evidence_ledger}}

CONTEXT:
{{context}}
""".format(
    compute_missing_policy_line=_COMPUTE_MISSING_POLICY_LINE,
    _extraction_zero_policy=(
        "If QUERY_STATE.missing_data_policy is `zero_if_not_explicit` and "
        "the required slot remains ungrounded, output @@ANSWER: 0. "
        "If `inapplicable_explain` and CONTEXT supports inapplicability, "
        "explain with citations instead of insufficient evidence."
    ),
    _finance_constraint_codes=_FINANCE_CONSTRAINT_CODES,
)


# Reflection prompt. Audits the draft against CONTEXT + EVIDENCE_LEDGER.
# The prior version added verbatim-claim / formula-identity exemptions to
# cover for synthesis's term-equivalence gap; with the synthesis prompt now
# encoding synonyms (rule 3 above), the exemptions are no longer needed.
# Reflection's job is narrowed to four checks that do NOT pile up false-FAILs
# on honest answers.
REFLECTION_PROMPT = """
Audit ANSWER against QUERY + CONTEXT + EVIDENCE_LEDGER. Default to PASS
unless a check below finds a concrete defect cited from QUERY_STATE /
EVIDENCE_LEDGER / CONTEXT. Record each finding under `checks_performed`
as a short free-text line.

Honest-abstain rule: if ANSWER is exactly `@@ANSWER: insufficient evidence`,
PASS when CONTEXT genuinely contains no candidate for the queried fact;
FAIL with `unnecessary_abstain` when a candidate IS present in CONTEXT.

Checks:
(A) Arithmetic & operand identity (compute queries only).
    - Recompute from EVIDENCE_LEDGER or CONTEXT values; if the recomputed
      value differs from the ANSWER's value beyond the requested rounding,
      FAIL with `arithmetic_check: fail` and include the recomputed value.
    - Standard line-item synonyms (capex ≡ purchases of PP&E; revenue ≡
      net sales; cost of revenue ≡ cost of sales; net income ≡ net earnings;
      operating income ≡ income from operations; gross profit ≡ gross
      margin in dollars) are NOT operand mismatches.
(B) Enumeration coverage. For extract/list/"what drove X" queries, the
    answer must report all grounded items present in CONTEXT for the
    queried scope. Missing items present in CONTEXT → `incomplete_enumeration`.
(C) Intent alignment. Concept substitution (answering an adjacent question)
    → FAIL. Answer that reports the queried subject from CONTEXT, even
    when using a synonym from check (A), is intent-aligned and PASS.
(D) Citation & format.
    - Cross-company or cross-period citations when same-company/same-period
      evidence is present in CONTEXT → FAIL with `fabricated_citation`.
    - Missing inline citations, multiple `@@ANSWER:` prefixes, or
      range/approximate values where the query requested exact → FAIL.
    - A numeric or named-entity value in ANSWER that does not appear in
      any cited chunk in CONTEXT → FAIL with `fabricated_claim`. Trivial
      reformatting (commas, units, parentheses for negatives) and synonym
      use under check (A) are allowed.

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
Refine the DRAFT answer using CRITIQUE. CONTEXT is the source of truth;
EVIDENCE_LEDGER is an advisory hint.

Rules:
1. Output JSON with `final_answer` starting with `@@ANSWER:`.
2. Keep all factual/numeric claims cited inline as `[[Title, Page X, Chunk Y]]`.
3. Apply minimal edits; preserve query type (extract/compute/boolean/list).
4. Keep the DRAFT's grounded conclusion unless CRITIQUE proves it incorrect.
5. Match the requested numeric format (unit, precision, rounding).
6. Compute: use operands present in EVIDENCE_LEDGER or CONTEXT; recompute
   at full precision and round only at the final step.
7. {compute_missing_policy_line}
8. Do not introduce new numeric values, units, periods, or named entities
   not present in CONTEXT or EVIDENCE_LEDGER.
9. If DRAFT already matches a calculator-derived numeric result, preserve it.
10. No meta or alternatives.

QUERY: {{query}}
QUERY_STATE: {{query_state}}
EVIDENCE_LEDGER: {{evidence_ledger}}
CONTEXT: {{context}}
DRAFT: {{draft}}
CRITIQUE: {{critique}}
""".format(
    compute_missing_policy_line=_COMPUTE_MISSING_POLICY_LINE,
)

REFINEMENT_RETRY_PROMPT = """
Previous refinement output was invalid: {error}
Output ONLY one JSON object with exactly this schema:
{{"final_answer":"@@ANSWER: ..."}}
Do not include alternatives, multiple @@ANSWER prefixes, or prose outside JSON.
PREVIOUS_OUTPUT: {previous_output}
"""
