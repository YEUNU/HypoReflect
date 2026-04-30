"""Shared, model-agnostic prompts for optional agentic query handling."""

AGENTIC_PLAN_PROMPT = """
Create an execution plan for financial QA retrieval.

Task:
- Rewrite the query into 1-3 high-precision retrieval queries.

Rules:
1. Preserve original meaning exactly.
2. Preserve company/entity and period tokens when present.
3. Keep metric/line-item terms explicit.
4. Do not invent new entities, periods, or assumptions.
5. Prefer statement-aware wording when relevant (income statement, balance sheet, cash flow statement, notes).

QUERY: {query}
"""

AGENTIC_PLAN_PROMPT_GENERIC = """
Create an execution plan for open-domain retrieval.

Task:
- Rewrite the query into 1-3 high-precision retrieval queries.

Rules:
1. Preserve original meaning exactly.
2. Preserve named entities, dates, and comparison targets when present.
3. Keep relation words explicit.
4. Do not invent new entities, dates, or assumptions.
5. Prefer concise variants that improve recall without changing intent.

QUERY: {query}
"""

AGENTIC_PLAN_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{"queries": ["..."]}
"""

AGENTIC_SYNTHESIS_PROMPT = """
Answer the question using ONLY the provided evidence context.

Output format (exact):
@@ANSWER: <final answer or insufficient evidence>
@@EVIDENCE:
- [[Doc, Page X, Chunk Y]] <supporting fact>

Rules:
1. Use only grounded evidence from CONTEXT.
2. If exact required operands are missing for a compute question, output insufficient evidence.
3. If multiple candidate values conflict, prefer the value with clearer statement-level grounding and citation.
4. Keep numeric strings exact; do not rescale units.

QUESTION:
{query}

CONTEXT:
{context}
"""

AGENTIC_REFLECTION_PROMPT = """
Audit the draft answer against the context.

Return PASS when grounded and directly answers the question.
Return FAIL when evidence is missing/conflicting or the answer is unsupported.

QUESTION:
{query}

DRAFT_ANSWER:
{answer}

CONTEXT:
{context}
"""

AGENTIC_REFLECTION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{"verdict":"PASS|FAIL","issues":["..."],"revised_answer":"..."}
"""
