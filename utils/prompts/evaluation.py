BENCHMARK_MATH_FORMAT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

BENCHMARK_MCQ_JSON_FORMAT_INSTRUCTION = (
    'Please show your choice in the answer field with only the choice letter, '
    'e.g., "answer": "C".'
)

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
### Task: Evaluate if the Model Prediction is correct based on the Ground Truth Answer.

**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

### Instructions:
1. The Model Prediction may contain step-by-step reasoning. Locate the FINAL answer:
   it is typically enclosed in \\boxed{{...}}, follows "Final Answer:", or follows the
   last "@@ANSWER:" marker. Compare ONLY that final answer against the Ground Truth
   — ignore intermediate reasoning steps and worked-out arithmetic.
2. Financial values must match numerically; equivalent unit scaling (million/billion/M) and formatting differences ($, commas, %) are acceptable.
3. If the final answer provides the correct factual/financial information as the Ground Truth, score it 1.0 (CORRECT).
4. If the final answer is factually wrong, contradicts the Ground Truth, or provides an incorrect value, score it 0.0 (INCORRECT).
5. Minor wording differences are okay as long as the core answer is the same.
6. If the question does not specify precision, treat equivalent values under standard rounding as correct.
7. Treat "Insufficient evidence." (or equivalent abstention) as 0.0 only when the Ground Truth contains a substantive answer; if the Ground Truth itself reflects "no value / 0 / not disclosed", an abstention may be acceptable.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "reason": "brief explanation"}}
"""

FINANCEBENCH_HALLUCINATION_PROMPT = """
### Task: Determine whether the Model Prediction is hallucinated against the Ground Truth.

**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

### Instructions:
1. The Model Prediction may contain step-by-step reasoning. Locate the FINAL answer
   (\\boxed{{...}}, after "Final Answer:", or after the last "@@ANSWER:") and judge
   hallucination on that final answer only.
2. Return hallucination=1.0 if the final answer asserts wrong/conflicting factual or numeric content.
3. Return hallucination=0.0 if the final answer is factually consistent with the Ground Truth.
4. If the final answer is exactly "insufficient evidence" (or equivalent non-answer), return hallucination=0.0.
5. Allow equivalent unit scaling/format differences when values are the same.

Respond ONLY in JSON format:
{{"hallucination": 1.0 or 0.0, "reason": "brief explanation"}}
"""
