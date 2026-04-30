_FINANCE_CONSTRAINT_CODES = (
    "C1 entity/period match, "
    "C2 source_anchor + primary statement priority, "
    "C3 placeholder/boilerplate is non-evidence, "
    "C4 exact numeric requests need exact values, "
    "C5 if ungrounded keep slot missing"
)

_EXTRACTION_CANONICAL_RULES = (
    "Keep `value` verbatim from CONTEXT (no paraphrase, conversion, or abbreviation); "
    "if citation spans multiple years, select only the value tied to slot.period; "
    "preserve accounting notation exactly (e.g., (123), -123, $1,234)."
)

_COMPUTE_MISSING_POLICY_LINE = (
    "For compute, if any required slot remains missing or conflicting, output @@ANSWER: insufficient evidence."
)
