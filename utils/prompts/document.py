OCR_PROMPT = """
Transcribe this page using Markdown. Rules:
1. Use Markdown for all text and tables.
2. Describe images, charts, and diagrams in detail as plain text.
"""

PAGE_SUMMARY_PROMPT = """
Summarize this page.
Rules:
1. Provide exactly one sentence, maximum 35 words.
2. Use only facts explicitly present on this page; do not infer, compute, aggregate, or normalize values.
3. If you mention numbers, copy exact value strings with unit/sign/decimal/currency/percent as written (no rounding or conversion).
4. If this page is mostly table-of-contents, exhibit index, signatures, or legal boilerplate, summarize only that structural fact and avoid financial claims.
5. Do not use filler words like "This page describes...".

TEXT:
{text}
"""

GROUP_SUMMARY_PROMPT = """
Synthesize these page summaries into one unified group summary.
Rules:
1. Provide exactly 2 sentences, maximum 60 words total.
2. Use only information present in the page summaries; do not introduce new facts or derived metrics.
3. Focus on the common thread connecting all pages.
4. Preserve key technical terms and any cited numeric strings exactly as written (no rounding or unit conversion).
5. If summaries are mostly structural/boilerplate, state that directly and avoid financial extrapolation.

PAGE SUMMARIES:
{page_summaries}
"""


ROLLING_SUMMARY_PROMPT = """
You are an expert technical document synthesizer.
Below is a 'Current Summary' of the preceding parts of a document, followed by 'New Content' from the next section.
Create a new, integrated, and dense technical summary that merges both.
Rules:
1. Keep entity/period/metric terms explicit.
2. If numbers are included, preserve exact value strings (unit/sign/decimal/currency/percent).
3. Do not convert units or round figures.

### Current Summary:
{running_summary}

### New Content:
{new_content}

New Integrated Summary:
"""

GLOBAL_SUMMARY_PROMPT = """
Analyze the provided page summaries to generate document-level metadata.
Rules:
1. Summary: Provide 3-5 concise sentences covering the entire document.
2. Keywords: Extract 5-7 specific technical terms.

PAGES:
{text}
"""

GLOBAL_SUMMARY_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"title": "String", "summary": "String", "keywords": []}}
"""

TABLE_TO_TEXT_PROMPT = """
Convert this table into plain text sentences.
Rules:
1. Each row must become exactly one natural language sentence.
2. Keep row/column labels explicit so metric + period/entity remain clear.
3. Copy numeric values exactly as written (sign, commas, decimals, %, currency, parentheses).
4. Do not round, rescale, or convert units.
5. Keep each sentence concise (prefer under 25 words), but do not drop row labels or values.
6. Output ONLY the plain text sentences, one per line.
"""
