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
1. Compare the Model Prediction with the Ground Truth Answer.
2. Financial values must match numerically; equivalent unit scaling (million/billion/M) and formatting differences ($, commas, %) are acceptable.
3. If the Model Prediction provides the correct factual/financial information as the Ground Truth, score it 1.0 (CORRECT).
4. If the Model Prediction is factually wrong, contradicts the Ground Truth, or provides an incorrect value, score it 0.0 (INCORRECT).
5. Minor wording differences are okay as long as the core answer is the same.
6. If the question does not specify precision, treat equivalent values under standard rounding as correct.

Respond ONLY in JSON format:
{{"score": 1.0 or 0.0, "reason": "brief explanation"}}
"""

FINANCEBENCH_HALLUCINATION_PROMPT = """
### Task: Determine whether the Model Prediction is hallucinated against the Ground Truth.

**Question:** {query}
**Ground Truth Answer:** {ground_truth}
**Model Prediction:** {response}

### Instructions:
1. Judge hallucination by comparing Model Prediction with Ground Truth.
2. Return hallucination=1.0 if the prediction asserts wrong/conflicting factual or numeric content.
3. Return hallucination=0.0 if the prediction is factually consistent with the Ground Truth.
4. If prediction is exactly "insufficient evidence" (or equivalent non-answer), return hallucination=0.0.
5. Allow equivalent unit scaling/format differences when values are the same.

Respond ONLY in JSON format:
{{"hallucination": 1.0 or 0.0, "reason": "brief explanation"}}
"""
