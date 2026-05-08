QUERY_CATEGORIZATION_PROMPT = """
Analyze the provided query and its ground truth evidence to categorize it for an ablation study.
Categories:
1. "Table": The answer is primarily located in a table or requires extracting specific numerical/categorical data from tabular structures.
2. "Global": The query asks about the document as a whole, trends across multiple years, or high-level summaries.
3. "Multi-hop": The query requires connecting 2 or more distinct pieces of information that are not adjacent in the text.
4. "Local": Simple fact extraction from a single, contiguous paragraph.

Query: {query}
Evidence: {evidence}

Respond in the following format:
CATEGORY: [Category Name]
REASON: [Short explanation]
"""

FINANCEBENCH_JUDGE_PROMPT = """
### Task: Score the Model Prediction on (a) correctness vs Ground Truth and
(b) hallucination — with a SINGLE LLM call so the two judgements are
internally consistent.

**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

### Instructions
1. Locate the FINAL answer in the Model Prediction (typically inside
   \\boxed{{...}}, after "Final Answer:", or after the last "@@ANSWER:" marker).
   Judge on that final answer only — ignore intermediate reasoning and
   worked-out arithmetic.
2. Allow equivalent unit scaling (million/billion/M) and formatting
   differences ($, commas, %, rounding) when the underlying value matches.
3. score:
   - 1.0 if the final answer conveys the same factual/financial content
     as the Ground Truth (minor wording differences ok).
   - 0.0 if the final answer is factually wrong, contradicts the Ground
     Truth, or provides an incorrect value.
   - "Insufficient evidence" / abstention is 0.0 when Ground Truth contains
     a substantive answer; if Ground Truth itself is "no value / 0 / not
     disclosed", an abstention is acceptable (1.0).
4. hallucination:
   - 1.0 if the final answer asserts wrong/conflicting factual or numeric
     content.
   - 0.0 if the final answer is factually consistent with the Ground
     Truth, or is an honest abstention ("insufficient evidence" / non-
     answer). An abstention is NOT a hallucination, even when the Ground
     Truth has a substantive answer.
5. Internal consistency: hallucination=1.0 implies score=0.0. score=1.0
   implies hallucination=0.0. score=0.0 with hallucination=0.0 is the
   honest-abstain case.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "hallucination": 1.0 or 0.0, "reason": "brief explanation covering both judgements"}}
"""
