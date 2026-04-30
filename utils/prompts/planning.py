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

PLANNING_PROMPT_GENERIC = """
Create a concise, executable retrieval plan for the query.
Rules:
1. First identify query type: extract|boolean|list|compute.
2. Extract key constraints: named entities, comparison targets, bridge clues, dates, and target relation/attribute.
3. Decompose into required evidence hops; for bridge questions, retrieve the intermediate entity before the final attribute.
4. For each step, specify what evidence to retrieve and how it will reduce uncertainty.
5. Prefer article/title/entity matches first, then supporting relation passages.
6. Stop as soon as the answer is directly grounded by citation-backed evidence.
7. If evidence conflicts, mark the unresolved hop and add one focused retrieval step for it.
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

PLANNING_FILTER_PROMPT_GENERIC = """
Create evidence filtering policy JSON for the query and retrieval plan.
Goal: define what evidence should be kept/dropped during ledger construction.
Rules:
1. Keep policy compact and operational.
2. Use query constraints first: named entities, dates, comparison targets, bridge intermediates, and target relation.
3. Specify strictness for entity/date matching; do not invent source anchors.
4. Prefer passages that directly identify bridge entities or answer-bearing facts.
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

AGENT_EXECUTION_SYSTEM_PROMPT = """
You are a financial filing retriever.
Goal: fill missing query slots with grounded evidence.
Constraint codes: <<FINANCE_CONSTRAINT_CODES>>.
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
QUERY_STATE:
{query_state}
MISSING_SLOTS:
{missing_slots}
CURRENT CONTEXT:
{context}
""".replace("<<FINANCE_CONSTRAINT_CODES>>", _FINANCE_CONSTRAINT_CODES)

AGENT_EXECUTION_SYSTEM_PROMPT_GENERIC = """
You are an open-domain evidence retriever.
Goal: fill missing query slots and bridge entities with grounded evidence.
Rules:
1. Use only `graph_search` or `calculator`; call at most one tool this turn.
2. Use `graph_search` for evidence retrieval; use `calculator` only for arithmetic.
3. Prioritize explicit named entities, comparison targets, and bridge entities from QUERY_STATE + MISSING_SLOTS.
4. For bridge questions, retrieve the intermediate person/work/entity first, then retrieve the final target attribute or relation.
5. Do not invent dates, document types, or statement anchors that are absent from the query.
6. Treat off-topic or weakly related evidence as non-evidence and continue retrieval.
7. Stop when the answer is directly supported by grounded evidence or the remaining gap is explicit.
QUERY_STATE:
{query_state}
MISSING_SLOTS:
{missing_slots}
CURRENT CONTEXT:
{context}
"""
