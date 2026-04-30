import logging

from core.config import RAGConfig
from models.hyporeflect.stages.execution_parts import (
    CalculatorSupport,
    ContextSupport,
    EvidenceSupport,
    EntitySupport,
    ExpansionLoopState,
    ForcedSynthesisSupport,
    OverrideSupport,
    QueryStateSupport,
    ResidualSupport,
    RuntimeSupport,
    SearchSupport,
    SlotSupport,
    SynthesisSupport,
    ToolCallsSupport,
)


logger = logging.getLogger(__name__)


class ExecutionHandler(
    ResidualSupport,
    QueryStateSupport,
    SlotSupport,
    EntitySupport,
    CalculatorSupport,
    ContextSupport,
    EvidenceSupport,
    ToolCallsSupport,
    OverrideSupport,
    SearchSupport,
    SynthesisSupport,
    ForcedSynthesisSupport,
    RuntimeSupport,
):
    _VALID_ANSWER_TYPES = {"extract", "compute", "boolean", "list"}
    _VALID_MISSING_DATA_POLICIES = {
        "insufficient",
        "zero_if_not_explicit",
        "inapplicable_explain",
    }
    _VALID_SLOT_CONFLICT_STRATEGIES = {
        "best_supported",
        "keep_missing_on_tie",
    }
    _QUERY_STATEMENT_ANCHOR_TERMS = (
        "balance sheet",
        "statement of financial position",
        "income statement",
        "statement of income",
        "p&l",
        "cash flow statement",
        "statement of cash flows",
    )
    _OPEN_DOMAIN_ENTITY_STOPWORDS = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "among",
        "between",
        "compare",
        "comparison",
        "list",
        "name",
    }
    _INSUFFICIENT_ANSWER_MARKERS = (
        "insufficient evidence",
        "cannot be determined",
        "not possible to determine",
        "cannot answer with the available",
        "does not contain",
        "not available in the context",
    )
    _ZERO_POLICY_ANSWERS = {"@@answer: 0", "@@answer: 0.0", "@@answer: 0.00"}
    _CAPEX_RELAXED_GROUNDING_MARKERS = (
        "purchases of property",
        "capital expenditure",
        "capital expenditures",
        "pp&e",
    )
    _CAPEX_AMOUNT_MARKERS = (
        "capital expenditure",
        "capital expenditures",
        "purchases of property",
        "purchases of pp&e",
        "additions to property and equipment",
        "additions to pp&e",
        "property, plant and equipment",
        "property and equipment",
    )
    _CAPEX_RATIO_SPAN_MARKERS = (
        "as a percentage",
        "% of net revenues",
        "% of net revenue",
        "% of revenue",
        "percent of net revenues",
        "percent of revenue",
    )
    _GENERIC_BOOTSTRAP_METRIC_TOKENS = {
        "income",
        "capital",
        "assets",
        "liabilities",
        "shareholders",
        "shareowners",
        "total",
        "metric",
        "change",
        "business",
    }
    _POST_SYNTHESIS_OVERRIDES = (
        (
            "operating_margin_override",
            "_build_operating_margin_driver_override_answer",
            "execution_operating_margin_driver_override",
            "operating_margin_driver_rule",
        ),
        (
            "segment_drag_override",
            "_build_segment_drag_override_answer",
            "execution_segment_drag_override",
            "segment_drag_rule",
        ),
        (
            "debt_securities_override",
            "_build_debt_securities_override_answer",
            "execution_debt_securities_override",
            "debt_securities_listing_rule",
        ),
        (
            "quick_ratio_override",
            "_build_quick_ratio_health_override_answer",
            "execution_quick_ratio_override",
            "quick_ratio_health_rule",
        ),
        (
            "capital_intensity_override",
            "_build_capital_intensity_override_answer",
            "execution_capital_intensity_override",
            "capital_intensity_ratio_rule",
        ),
        (
            "dividend_override",
            "_build_dividend_stability_override_answer",
            "execution_dividend_stability_override",
            "dividend_stability_rule",
        ),
    )

    def __init__(self, llm, grag, stage_model: str = ""):
        self.llm = llm
        self.grag = grag
        self.stage_model = stage_model or RAGConfig.EXECUTION_MODEL
        self.context_node_budget = 36
        self.context_char_budget = 4200
