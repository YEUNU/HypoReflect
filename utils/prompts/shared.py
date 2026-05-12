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

# Permissive: missing slots block compute only when CONTEXT also has no candidate.
# Prior wording forced abstain on any missing slot, which over-rejected answers
# that retrieval did surface but the slot extractor failed to bind.
_COMPUTE_MISSING_POLICY_LINE = (
    "For compute: if a required operand has no candidate in CONTEXT or EVIDENCE_LEDGER, output @@ANSWER: insufficient evidence; "
    "otherwise use the candidate from CONTEXT even if EVIDENCE_LEDGER did not bind it."
)
