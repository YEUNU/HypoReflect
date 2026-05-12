from .shared import _FINANCE_CONSTRAINT_CODES


PERCEPTION_PROMPT = """
Classify the user query for agent routing.
Return:
1. complexity: "simple" or "complex"

Guidelines:
- Use "complex" if the query likely requires multi-step retrieval, multi-hop evidence, or numerical reasoning.
- Otherwise use "simple".

QUERY: {query}
"""

PERCEPTION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"complexity":"simple|complex","reason":"..."}}
"""

PERCEPTION_RETRY_PROMPT = """
Previous perception output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"complexity":"simple|complex","reason":"..."}}
PREVIOUS_OUTPUT: {previous_output}
"""

PLANNING_PROMPT = """
Create a concise, executable retrieval plan for the query.
Rules:
1. First identify query type: extract|compute|boolean|list.
2. Extract key constraints: company/entity, year/period, document type (10-K/10-Q), target metric, source_anchor (if explicit).
3. Decompose into required evidence slots; for compute queries use primitive operands (not only derived metric labels).
4. For each slot, specify what evidence to retrieve (statement/note section + line-item/indicator + period column).
5. Set retrieval priority: primary statements/note tables first, narrative text later.
6. Define stop condition: stop immediately when all required slots are citation-grounded and constraint-consistent.
7. Define conflict handling: if slot values conflict, mark unresolved and plan one additional retrieval focus for that slot.
8. Keep output to 3-5 numbered steps and include a final verification step.
Output format for each step:
- Step N: objective | target slots | evidence target | done condition
QUERY: {query}
CONTEXT: {context}
"""

PLANNING_FILTER_PROMPT = """
Create evidence filtering policy JSON for the query and retrieval plan.
Goal: define what evidence should be kept/dropped during ledger construction.
Rules:
1. Keep policy compact and operational.
2. Use query constraints first: entity, period, metric, source_anchor.
3. Specify strictness for entity/period/source_anchor matching.
4. List preferred statement markers and disallowed patterns.
5. Define conflict strategy for multiple candidates per slot.
QUERY: {query}
PLAN: {plan}
"""

PLANNING_FILTER_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{
  "must_match": {"entity": true, "period": true, "source_anchor": "strict|soft|none"},
  "preferred_markers": [],
  "disallowed_patterns": [],
  "slot_conflict_strategy": "best_supported|keep_missing_on_tie"
}
"""

PLANNING_FILTER_RETRY_PROMPT = """
Previous planning filter output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"must_match":{{"entity":true,"period":true,"source_anchor":"strict|soft|none"}},"preferred_markers":[],"disallowed_patterns":[],"slot_conflict_strategy":"best_supported|keep_missing_on_tie"}}
PREVIOUS_OUTPUT: {previous_output}
"""

PLANNING_MERGED_PROMPT = """
Create a concise retrieval plan AND a filtering policy for the query in
ONE pass. Output JSON only.

Plan rules:
1. Identify query type (extract|compute|boolean|list).
2. Extract key constraints: company/entity, year/period, document type
   (10-K/10-Q), target metric, source_anchor.
3. Decompose into required evidence slots; for compute queries use
   primitive operands.
4. For each slot, specify what evidence to retrieve (statement/note
   section + line-item + period column).
5. Set retrieval priority: primary statements first, narrative later.
6. Stop condition: all required slots are citation-grounded.
7. Conflict handling: if slot values conflict, mark unresolved + one
   focused retry.
8. 3-5 numbered steps + a final verification step.

Filter policy rules:
1. Use query constraints first: entity, period, metric, source_anchor.
2. Specify strictness for entity/period/source_anchor.
3. List preferred statement markers and disallowed patterns.
4. Define conflict strategy.

QUERY: {query}
CONTEXT: {context}
"""

PLANNING_MERGED_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{
  "plan": "Step 1: ...\\nStep 2: ...\\nStep 3: ...",
  "filter_policy": {
    "must_match": {"entity": true, "period": true, "source_anchor": "strict|soft|none"},
    "preferred_markers": [],
    "disallowed_patterns": [],
    "slot_conflict_strategy": "best_supported|keep_missing_on_tie"
  }
}
"""

PLANNING_MERGED_RETRY_PROMPT = """
Previous merged-planning output was invalid.
Error: {error}
Output ONLY one JSON object with exactly this schema:
{{"plan":"Step 1: ...\\nStep 2: ...","filter_policy":{{"must_match":{{"entity":true,"period":true,"source_anchor":"strict|soft|none"}},"preferred_markers":[],"disallowed_patterns":[],"slot_conflict_strategy":"best_supported|keep_missing_on_tie"}}}}
PREVIOUS_OUTPUT: {previous_output}
"""

AGENT_EXECUTION_SYSTEM_PROMPT = """
You are a financial filing retriever.
Goal: fill missing query slots with grounded evidence.
Constraint codes: <<FINANCE_CONSTRAINT_CODES>>.

Each turn, BEFORE deciding the next action, scan CURRENT CONTEXT for any
value that matches a MISSING_SLOT. Emit each grounded find as an EVIDENCE
line in the format below, then output the tool call. EVIDENCE lines and
the tool_call may appear in any order. Lines that fail to parse are
silently skipped — emit best-effort.

EVIDENCE line format (one pair per line, tolerant of partial output):
EVIDENCE: value=<verbatim value from CONTEXT> | citation=[[Doc, Page X, Chunk Y]] | metric=<slot.metric you intend this for>

Rules:
1. Use only `graph_search` or `calculator`; call at most one tool this turn.
2. Use `graph_search` for evidence retrieval; use `calculator` only for arithmetic.
3. If QUERY_STATE.entity is non-empty, graph_search args `entities` MUST include that company/entity token as the first item.
3a. If QUERY_STATE.entity is empty, do not invent generic company placeholders.
4. Never use metric-only `entities` values (for example "restructuring costs" alone); include company/entity + period + metric together when possible.
5. Target MISSING_SLOTS directly; for compute, collect primitive operands before derived metrics.
6. For compute queries: when required slots are grounded, call `calculator` with one explicit expression; use precision from QUERY_STATE.rounding when available.
7. Treat C1-violating evidence as non-evidence and continue retrieval.
8. Stop when all required slots are grounded and computation is resolved.
9. TOOL_HISTORY shows prior tool calls and outcomes (new_entries, reject_reasons). MANDATORY rules:
   (a) The next graph_search MUST contain at least one `entities` token that did NOT appear in any prior call's `entities` set. Repeating the same token bag with `top_k`-only changes is forbidden.
   (b) When prior calls returned 0 new ledger entries, prefer adding alternate line-item names that the SAME line-item could appear under (e.g., the formula-name, the GAAP caption, the table-row label, the note-table label). Cycle through: query-literal name → standard accounting caption → note-table label.
   (c) When prior calls hit `value_period_mismatch` or `citation_period_mismatch`, restate the period in a different form on the next attempt (FY token, calendar-year token, "year ended <date>"); do NOT just repeat the period string.
   (d) When prior calls hit `source_anchor_mismatch`, switch the anchor (e.g., income statement ↔ cash flow statement ↔ balance sheet) on the next attempt.
   (e) If three consecutive calls failed similarly, abandon this slot and call calculator on partially-grounded operands or move to the next missing slot.

QUERY_STATE:
{query_state}
MISSING_SLOTS:
{missing_slots}
TOOL_HISTORY:
{tool_history}
CURRENT CONTEXT:
{context}
""".replace("<<FINANCE_CONSTRAINT_CODES>>", _FINANCE_CONSTRAINT_CODES)
