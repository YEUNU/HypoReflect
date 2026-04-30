from .shared import (
    _EXTRACTION_CANONICAL_RULES,
    _FINANCE_CONSTRAINT_CODES,
)


QUERY_STATE_PROMPT = """
Extract compact QUERY_STATE JSON for finance QA.
Fields: entity, period, metric, source_anchor(cash flow statement|income statement|balance sheet|note table|null), answer_type(extract|compute|boolean|list), required_slots, unit, rounding, missing_data_policy(insufficient|zero_if_not_explicit|inapplicable_explain).
Rules:
1. required_slots must be a JSON array of slot objects for ALL answer types (no plain strings).
2. Top-level entity must be the target company/entity in QUERY (not a metric phrase).
3. Every slot.entity must match top-level entity unless QUERY explicitly asks a sub-entity/segment; never put metric terms in slot.entity.
4. For compute, required_slots must be primitive operands; do not use only a derived metric slot.
5. For multi-period compute queries, split into one atomic slot per period/operand.
6. For non-compute queries, required_slots must still be non-empty and minimally include at least one citation-checkable claim slot.
7. If period is not explicit in QUERY, keep period as empty string in both top-level and slots instead of inventing one.
8. If QUERY explicitly names statement anchor(s) (balance sheet/income statement/cash flow), set source_anchor at top-level or in every required slot (prefer slot-level for multi-statement compute).
9. Do not output dict-strings, serialized JSON strings, or formulas in slot fields.
9a. Entity extraction must be specific: if company/entity is not explicitly stated or inferable, set entity to empty string "". Do NOT use placeholders like "company", "the company", "entity", "organization", or similar.
10. Set missing_data_policy from explicit query instruction only:
   - "zero_if_not_explicit" when query explicitly says missing/not explicitly outlined => return 0.
   - "inapplicable_explain" only for non-compute queries when query explicitly asks to explain if metric is not useful/applicable.
   - compute queries must not use "inapplicable_explain"; use "insufficient" unless query explicitly says zero_if_not_explicit.
   - otherwise "insufficient".
QUERY: {query}
"""

QUERY_STATE_PROMPT_GENERIC = """
Extract compact QUERY_STATE JSON for open-domain multi-hop QA.
Fields: entity, period, metric, source_anchor(null unless explicit), answer_type(extract|compute|boolean|list), required_slots, unit, rounding, missing_data_policy(insufficient|zero_if_not_explicit|inapplicable_explain).
Rules:
1. required_slots must be a JSON array of atomic slot objects for ALL answer types.
2. Top-level entity may be empty string or a primary anchor entity; do not collapse multiple entities into one slot.
3. For comparison questions, include one slot per comparison target entity.
4. For bridge questions, include slots for explicit entities/work titles needed to identify the intermediate entity and the final answer-bearing fact.
5. metric may be a short relation/attribute label such as nationality, spouse, government position, birthplace, or portrayed role.
6. If period/date is not explicit in QUERY, keep period as empty string instead of inventing one.
7. Set source_anchor to null unless the query explicitly requests a source type.
8. required_slots for extract/boolean/list must remain non-empty and citation-checkable.
9. Do not output dict-strings, serialized JSON strings, or prose in slot fields.
10. missing_data_policy should be "insufficient" unless the query explicitly requests another supported policy.
QUERY: {query}
"""

QUERY_STATE_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"entity":"...","period":"...","metric":"...","source_anchor":null,"answer_type":"extract|compute|boolean|list","required_slots":[{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}}],"unit":null,"rounding":null,"missing_data_policy":"insufficient|zero_if_not_explicit|inapplicable_explain"}}
"""

QUERY_STATE_RETRY_PROMPT = """
Previous QUERY_STATE was invalid.
Errors: {errors}
Rewrite QUERY_STATE as strict JSON only.
required_slots must be a JSON array of atomic slot objects with concrete entity/period/metric/source_anchor fields.
missing_data_policy must be one of insufficient|zero_if_not_explicit|inapplicable_explain.
Never use generic placeholder entities like "company"/"entity"; use empty string if unknown.
Do not output dict-strings, formulas, or prose.
QUERY: {query}
PREVIOUS_OUTPUT: {previous_output}
"""

QUERY_STATE_REVIEW_PROMPT = """
Review and correct QUERY_STATE for the given query.
Rules:
1. Keep only query-grounded values.
2. Ensure entity is the target company/entity in the query.
2a. If query does not specify a company/entity, set entity to empty string "". Never use generic placeholders.
3. Every required_slots slot.entity must equal QUERY_STATE.entity unless QUERY explicitly asks a sub-entity/segment.
4. Set source_anchor only if explicitly requested in query; else null.
5. required_slots must be atomic slot objects (not strings), concrete, and citation-checkable.
6. For compute queries, required_slots must be primitive operands, not only final derived labels.
7. For extract/boolean/list, required_slots must not be empty.
8. Reject vague slots, embedded JSON/dict strings, or combined-period slot expressions.
9. Keep missing_data_policy query-grounded and valid (insufficient|zero_if_not_explicit|inapplicable_explain).
QUERY: {query}
DRAFT_QUERY_STATE: {draft_query_state}
"""

QUERY_STATE_REVIEW_PROMPT_GENERIC = """
Review and correct QUERY_STATE for the given query.
Rules:
1. Keep only query-grounded values.
2. Top-level entity may be empty string or a primary anchor entity; never use generic placeholders.
3. Do not collapse multiple comparison or bridge entities into a single slot.
4. Keep source_anchor null unless the query explicitly requests a source type.
5. required_slots must be atomic slot objects, concrete, and citation-checkable.
6. For comparison questions, keep one slot per comparison target entity.
7. For bridge questions, keep slots that preserve the intermediate entity/article needed to reach the answer.
8. Keep missing_data_policy query-grounded and valid.
QUERY: {query}
DRAFT_QUERY_STATE: {draft_query_state}
"""

EVIDENCE_LEDGER_PROMPT = """
Build an evidence ledger from CONTEXT for QUERY_STATE.
Constraint codes: <<FINANCE_CONSTRAINT_CODES>>.
Rules:
1. Keep only citation-grounded entries from CONTEXT.
2. Each entry must map to exactly one REQUIRED_SLOTS slot and one citation.
3. `slot` must be one REQUIRED_SLOTS slot object copied exactly (same entity/period/metric/source_anchor fields).
4. <<EXTRACTION_CANONICAL_RULES>>
5. Match slot metric qualifiers strictly (net vs gross, total/consolidated vs segment/product); non-matching variants must be rejected.
6. For company-level revenue/net-sales slots, keep consolidated primary income-statement totals; reject segment/geography/product values unless explicitly requested.
7. Reject guidance/forward-looking values for extract/list questions.
8. Reject placeholders, slot-label echoes, and empty/non-evidence values.
9. For numeric/ratio slots, accept only values with numeric evidence text; else keep slot missing.
10. Entity/period mismatch to QUERY_STATE or slot is hard rejection; never map such evidence.
11. If multiple candidates exist for a slot, select one best-supported value using C1/C2 + exact line-item match + strongest citation support.
12. Return missing_slots as uncovered REQUIRED_SLOTS slot objects.
13. Do not mark a slot missing when at least one valid candidate remains after applying hard rejections.
14. Hard reject entries when citation title/document clearly belongs to another company/entity.
QUERY_STATE: {query_state}
FILTER_POLICY: {filter_policy}
CONTEXT: {context}
""".replace("<<FINANCE_CONSTRAINT_CODES>>", _FINANCE_CONSTRAINT_CODES).replace(
    "<<EXTRACTION_CANONICAL_RULES>>",
    _EXTRACTION_CANONICAL_RULES,
)

EVIDENCE_LEDGER_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"entries":[{{"slot":{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}}, "value":"...","citation":"[[Title, Page X, Chunk Y]]"}}],"missing_slots":[{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}}]}}
"""

CONTEXT_ATOMIZATION_PROMPT = """
Extract compact evidence atoms from context for QUERY_STATE.
Rules:
1. Keep citation-grounded atoms only.
2. Preserve exact value text from context (no conversion/reformatting).
3. Map each atom to required_slots by meaning and use required_slots slot objects in supports_slots.
4. For unresolved required slots, emit at least one candidate atom per slot when evidence exists.
5. If required_slots is non-empty and evidence exists, do not output 0-1 atoms by default.
6. Coverage target: at least min(3, required_slots_count) atoms, unless context has no citation-grounded evidence.
7. Reject atoms when citation title/document clearly belongs to a different company/entity than QUERY_STATE.
8. Drop boilerplate/non-evidentiary text.
QUERY_STATE: {query_state}
CONTEXT: {context}
"""

CONTEXT_ATOMIZATION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"atoms":[{{"atom_id":"a1","citation":"[[Title, Page X, Chunk Y]]","span":"...","supports_slots":[{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}}]}}]}}
"""

CONTEXT_ATOMIZATION_RETRY_PROMPT = """
Previous CONTEXT_ATOMIZATION output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"atoms":[{{"atom_id":"a1","citation":"[[Title, Page X, Chunk Y]]","span":"...","supports_slots":[{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}}]}}]}}
Use only citations that appear in CONTEXT.
PREVIOUS_OUTPUT: {previous_output}
"""

CONTEXT_PACKING_PROMPT = """
Select a minimal atom set that best covers required_slots under budget.
Rules:
1. Prioritize slot coverage, then minimize atom count.
2. Select at least one supporting atom per required slot when available.
3. If required_slots is non-empty, avoid single-atom collapse: selected_atom_ids should normally be >= min(2, required_slots_count).
4. If required slots remain uncovered and relevant atoms fit budget, add atoms before declaring missing.
5. missing_slots can be empty only when every required slot has at least one selected supporting atom.
6. Prefer atoms with clear entity/period/line-item alignment over generic narrative text.
7. Keep only atoms useful for final answer.
QUERY_STATE: {query_state}
BUDGET_CHARS: {budget_chars}
ATOMS: {atoms}
"""

CONTEXT_PACKING_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"selected_atom_ids":[],"slot_coverage":{{}},"missing_slots":[]}}
"""

CONTEXT_PACKING_RETRY_PROMPT = """
Previous CONTEXT_PACKING output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"selected_atom_ids":[],"slot_coverage":{{}},"missing_slots":[]}}
selected_atom_ids must use only ATOMS.atom_id values.
PREVIOUS_OUTPUT: {previous_output}
"""

EVIDENCE_LEDGER_RETRY_PROMPT = """
Previous EVIDENCE_LEDGER output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"entries":[{{"slot":{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}},"value":"...","citation":"[[Title, Page X, Chunk Y]]"}}],"missing_slots":[{{"entity":"...","period":"...","metric":"...","source_anchor":"..."}}]}}
If you cannot map a slot, keep it in missing_slots and omit invalid entries.
PREVIOUS_OUTPUT: {previous_output}
"""

EVIDENCE_LEDGER_ZERO_RESCUE_PROMPT = """
Previous EVIDENCE_LEDGER response contained zero entries.
Re-read CONTEXT and recover REQUIRED_SLOTS with strict slot-by-slot extraction.
Rules:
1. For each required slot, emit one entry if any grounded candidate exists.
2. `slot` must exactly match one REQUIRED_SLOTS slot object.
3. Keep `value` verbatim from CONTEXT and use citations exactly as they appear in CONTEXT.
4. Leave slot in missing_slots only when no grounded candidate exists.
QUERY_STATE: {query_state}
FILTER_POLICY: {filter_policy}
CONTEXT: {context}
"""

ENTRY_GATE_PROMPT = """
Audit candidate evidence entries for slot validity.
Rules:
1. keep=true ONLY if value directly grounds slot(entity/period/metric/source_anchor) from the cited span.
2. For extract/boolean/list, reject forward-looking/guidance/expected/target/planned values.
3. If source_anchor is income statement/balance sheet/cash flow statement, reject narrative or non-statement evidence.
4. For company-level revenue/net-sales slots, keep consolidated company totals; reject segment/geography/product-only values.
4a. If slot.source_anchor is income statement and metric is revenue/net sales at company level, keep ONLY values grounded in consolidated statements of operations/income; reject segment tables even if numbers are plausible.
5. If uncertain, set keep=false.
QUERY_STATE: {query_state}
CANDIDATES: {candidates}
CONTEXT: {context}
"""

ENTRY_GATE_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"decisions":[{{"index":0,"keep":true,"reason":"..."}}]}}
"""

ENTRY_GATE_RETRY_PROMPT = """
Previous ENTRY_GATE output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"decisions":[{{"index":0,"keep":true,"reason":"..."}}]}}
PREVIOUS_OUTPUT: {previous_output}
"""

SLOT_CANDIDATE_VERIFIER_PROMPT = """
Select the best candidate per slot from provided SLOT_CANDIDATES.
Rules:
1. You must choose only from candidate_id values provided for each slot; never invent new values or citations.
2. Respect QUERY_STATE constraints first: entity, period, metric, source_anchor, and compute operand intent.
3. Prefer candidates with direct metric/period alignment and cleaner citation grounding.
4. If no candidate is clearly reliable for a slot, return verdict=unresolved for that slot.
5. Keep output concise and deterministic; do not include prose outside JSON.
QUERY: {query}
QUERY_STATE: {query_state}
SLOT_CANDIDATES: {slot_candidates}
CONTEXT: {context}
"""

SLOT_CANDIDATE_VERIFIER_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{
  "decisions": [
    {
      "slot_key": "normalized slot key",
      "selected_candidate_id": "c1|null",
      "verdict": "selected|unresolved",
      "confidence": 0.0,
      "reason": "short reason"
    }
  ]
}
"""

SLOT_CANDIDATE_VERIFIER_RETRY_PROMPT = """
Previous SLOT_CANDIDATE_VERIFIER output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"decisions":[{{"slot_key":"...","selected_candidate_id":"c1|null","verdict":"selected|unresolved","confidence":0.0,"reason":"..."}}]}}
Do not invent candidate_id values.
PREVIOUS_OUTPUT: {previous_output}
"""

MISSING_SLOT_RESCUE_PROMPT = """
Recover entries only for MISSING_SLOTS.
Rules:
1. Emit entries only for slots listed in MISSING_SLOTS.
2. `slot` must exactly match one slot in MISSING_SLOTS.
3. Use citation string EXACTLY from ALLOWED_CITATIONS.
4. Keep `value` as exact CONTEXT span (no conversion or abbreviation).
5. Do not emit entries for already-covered slots.
6. If slot cannot be grounded from context, leave it missing.
7. If prior entry failed due malformed citation format, repair citation only and preserve grounded value.
QUERY_STATE: {query_state}
MISSING_SLOTS: {missing_slots}
ALLOWED_CITATIONS: {allowed_citations}
CONTEXT: {context}
"""

CHAT_SYSTEM_FORMAT_INSTRUCTION = """
Rules:
1. Prefix the final answer with @@ANSWER:.
2. Cite key facts in [[Title, Page X, Chunk Y]].
3. Match numeric format requested by the query (unit/precision).
4. Do not guess or output audit/meta commentary.
"""

FINAL_ANSWER_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{"final_answer":"@@ANSWER: ..."}
"""

FINAL_ANSWER_RETRY_PROMPT = """
Previous {stage} output was invalid: {reason}
Output ONLY one JSON object with exactly this schema:
{{"final_answer":"@@ANSWER: ..."}}
Do not include alternatives, multiple @@ANSWER prefixes, or prose outside JSON.
PREVIOUS_OUTPUT: {previous_output}
"""

CALCULATION_PLAN_PROMPT = """
Build one arithmetic expression for the query using ONLY numbers from EVIDENCE_LEDGER values.
Rules:
1. Use only numbers and operators (+,-,*,/,**, parentheses).
2. Respect query formula exactly.
3. For average across periods, include explicit average in expression.
4. Do not invent values.
5. precision should follow QUERY_STATE.rounding when present, else null.
QUERY: {query}
QUERY_STATE: {query_state}
EVIDENCE_LEDGER: {evidence_ledger}
"""

CALCULATION_PLAN_RETRY_PROMPT = """
Previous calculation plan output was invalid.
Error: {error}
Output ONLY one JSON object: {{"expression":"...","precision":2|null}}
PREVIOUS_OUTPUT: {previous_output}
"""
