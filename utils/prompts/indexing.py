HOPRAG_PROMPT = """
Analyze this financial document chunk and generate hypothetical questions to enable multi-hop reasoning.
Rules:
1. Q- (Incoming): up to 3 questions that this chunk directly answers; use [] if the chunk lacks concrete answerable facts.
2. Q+ (Bridge): up to 3 follow-up questions that remain grounded in this chunk but need other chunks/docs; use [] if no high-quality bridge question exists.
3. Every produced question must be specific, answerable, and <= 22 words.
4. Every produced question must include at least one entity token and one metric/line-item token that appear in this chunk.
5. If a year/period token exists in this chunk, include it in each question.
6. Never use placeholders/meta phrases such as "document anchor", "what does the balance sheet show", or lists like "(balance sheet, income statement, cash flow statement, note table)".
7. Never fabricate unseen values, dates, entities, policies, or legal details.
8. If this chunk is mostly TOC/exhibits/signatures/boilerplate or numeric fragments with weak context, return shorter lists (or empty lists) rather than low-quality questions.
9. If the chunk contains computation cues (change, increase/decrease, ratio, margin, versus/prior period, multi-period values), produce at least 1 Q+ targeting the missing counterpart operand/period.
10. Dense Summary: exactly 1 sentence, maximum 35 words, grounded only in this chunk; preserve numeric strings exactly when present.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""

HOPRAG_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"summary": "concise informative summary", "q_minus": ["q1", "q2", "q3"], "q_plus": ["q1", "q2", "q3"]}}
"""

QUERY_REWRITE_PROMPT = """
Rewrite the query into finance-focused retrieval variants.
Rules:
1. Generate 1-3 high-precision variants preserving original meaning.
2. Detect constraint anchors from the original query: target company/entity token(s) and target period token(s) (year/FY/quarter/date).
3. Every variant must include the same target company/entity token(s) and target period token(s) when present; if they cannot be preserved, do not emit that variant.
4. Keep metric, numeric qualifiers, and formula definition unchanged.
5. Preserve exact tokens for symbols/segments/line items when present (e.g., MMM26, consumer segment).
6. Each variant must include at least one statement anchor term (balance sheet, income statement, cash flow statement, note table, PP&E, accounts receivable, inventory, debt securities).
7. Apply filing synonym normalization only when equivalent:
   - revenue ↔ net sales
   - capex ↔ purchases of property, plant and equipment (PP&E)
   - net PP&E ↔ property, plant and equipment — net
   - net AR ↔ trade accounts receivable, net
8. Do NOT introduce another company/year/period, unsupported assumptions, or special query syntax operators.
Original Query: {query}
"""

QUERY_REWRITE_PROMPT_GENERIC = """
Rewrite the query into domain-neutral retrieval variants.
Rules:
1. Generate 1-3 high-precision variants preserving original meaning.
2. Preserve named entities, dates, qualifiers, and comparison targets exactly when present.
3. Do not introduce domain-specific jargon that is absent from the original query.
4. Prefer lexical variants and disambiguating phrasing that improve recall without changing intent.
5. Do NOT introduce unsupported assumptions, extra entities, or special query syntax operators.
Original Query: {query}
"""

QUERY_REWRITE_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"positive_queries": []}}
"""

RERANKER_INSTRUCTION = (
    "Given a finance QA query, rank passages from SEC filings by direct answerability. "
    "Hard constraints: prioritize passages matching the exact target company/entity and year/period; strongly penalize company/period mismatches. "
    "Prefer primary financial statements and note tables containing required line items or formula operands. "
    "For extraction/list questions, prioritize passages that explicitly contain exact target tokens (symbols, segment names, note titles). "
    "Treat standard synonyms as equivalent (revenue=net sales, capex=purchases of PP&E, net PP&E=property plant and equipment net). "
    "Down-rank boilerplate legal/risk/exhibit text unless explicitly requested."
)

RERANKER_INSTRUCTION_GENERIC = (
    "Given a question, rank passages by direct answerability and evidential relevance. "
    "Hard constraints: prioritize passages matching the same entities, dates, and relation described in the query; "
    "strongly penalize entity or date mismatches. "
    "Prefer passages that contain explicit facts useful for multi-hop reasoning or direct comparison. "
    "Down-rank boilerplate, navigation text, and weakly related passages."
)

SEARCH_CONTINUATION_PROMPT = """
Decide whether retrieval should continue.
Decision rules:
1. Infer required evidence slots from QUERY. For compute queries, infer all primitive operands needed for the formula, not only the final derived metric.
2. Return "SUFFICIENT" only when all required slots are grounded in context with matching target company/entity and target period constraints.
3. For compute queries, evidence may come from multiple pages/documents; do not require a single-document hit.
4. Return "INSUFFICIENT" if any required slot is missing, ambiguous, conflicting, or tied to the wrong entity/period.
5. Prefer stopping as soon as slot coverage is complete; avoid unnecessary extra hops.
QUERY: {query}
CONTEXT: {context}
"""

SEARCH_CONTINUATION_PROMPT_GENERIC = """
Decide whether retrieval should continue.
Decision rules:
1. Infer the minimum evidence needed to answer QUERY directly.
2. Return "SUFFICIENT" only when the current context contains enough grounded evidence to answer the query.
3. Return "INSUFFICIENT" if a required entity, relation, comparison target, or date is missing, ambiguous, or conflicting.
4. Multi-hop questions may require evidence from multiple passages; do not require all evidence to come from one passage.
5. Prefer stopping as soon as the evidence is sufficient; avoid unnecessary extra hops.
QUERY: {query}
CONTEXT: {context}
"""

SEARCH_CONTINUATION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"decision": "SUFFICIENT"|"INSUFFICIENT", "next_focus": "..."}}
"""
