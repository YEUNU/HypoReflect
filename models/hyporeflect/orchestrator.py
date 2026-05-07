"""Five-stage reflective agent orchestrator (paper §3.2).

Sequences Perception -> Planning -> Execution -> Reflection -> Refinement.
Branches into a single-pass non-agentic path when RAG_AGENTIC_MODE=off (the
agentic-OFF retrieval baseline used in paper §4.4 ablation table).

Responsibilities split out from AgentService:
- Refinement ranking + R_max loop -> stages/refinement.RefinementOrchestrator
- Non-agentic prompt building -> _build_simple_answer_prompt
"""
import logging
import os
from typing import Any, Optional

from core.config import RAGConfig
from models.hyporeflect.state import AgentState
from models.hyporeflect.stages.refinement import RefinementOrchestrator


logger = logging.getLogger(__name__)


_ANSWER_PREFIX = "@@ANSWER:"


class Orchestrator:
    """Sequences the five-stage agentic pipeline; non-agentic baseline lives
    on the same class so the entry point matches paper §4.4 settings."""

    def __init__(
        self,
        *,
        llm,
        grag,
        perception,
        planning,
        execution,
        reflection,
        refinement,
    ):
        self.llm = llm
        self.grag = grag
        self.perception = perception
        self.planning = planning
        self.execution = execution
        self.reflection = reflection
        self.refinement = refinement
        self.refinement_loop = RefinementOrchestrator(
            refinement=refinement,
            reflection=reflection,
            execution=execution,
        )

    # ---------- formatting ----------
    @classmethod
    def _ensure_answer_prefix(cls, answer: str) -> str:
        text = str(answer or "")
        if _ANSWER_PREFIX not in text:
            return f"{_ANSWER_PREFIX} {text}"
        return text

    @staticmethod
    def _strip_format_instruction(query: str) -> str:
        """Remove the benchmark-format suffix that cli/benchmark.py may attach
        before sending the query to retrieval. The format instruction is
        answer-formatting noise when embedded for ANN search."""
        marker = "[Benchmark Output Format]"
        if marker in query:
            return query.split(marker, 1)[0].strip()
        return query

    @staticmethod
    def _build_unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_sources: list[dict[str, Any]] = []
        seen = set()
        for row in rows:
            doc = row.get("title") or row.get("doc") or "Unknown"
            page = row.get("page", 0)
            sent_id = row.get("sent_id", 0)
            key = (doc, page, sent_id)
            if key in seen:
                continue
            unique_sources.append(
                {
                    "doc": doc,
                    "page": page,
                    "text": row.get("text", ""),
                    "sent_id": sent_id,
                }
            )
            seen.add(key)
        return unique_sources

    # ---------- non-agentic baseline ----------
    @staticmethod
    def _build_simple_answer_prompt(context: str, user_query: str) -> str:
        # Agentic-OFF retrieval baseline path (paper §4.4). The mode isolates
        # retrieval quality, so the model must always attempt an answer from
        # CONTEXT — abstention is intentionally NOT offered. The lazy-answer
        # guard (paper §3.2.4 line 327) only lives in agentic-ON Reflection.
        return (
            "You are a financial analyst answering a question about an SEC filing.\n"
            "Use ONLY the passages in CONTEXT. Always produce a substantive answer; do not refuse or claim insufficient evidence.\n"
            "\n"
            "Output rules:\n"
            "1. Begin the answer with `@@ANSWER:` followed by the substantive answer on the same line.\n"
            "2. Cite every numeric or factual claim inline as `[[Title, Page X, Chunk Y]]` using the exact IDs printed in CONTEXT.\n"
            "3. For derived metrics (margin, ratio, growth, YoY change, average), compute from primitive operands present in CONTEXT and show one formula with substituted values, then the final value.\n"
            "4. Treat synonyms as equivalent: revenue = net sales; capex = purchases of PP&E; net PP&E = property plant and equipment net; net AR = trade accounts receivable net; FCF = CFO - capex.\n"
            "5. Operands often live in different statements (balance sheet / income statement / cash flow / notes). Combine them as needed.\n"
            "6. If the question asks whether a metric is true/false (yes/no), give the verdict first, then the supporting numbers.\n"
            "7. If the requested period or entity is only partially covered in CONTEXT, ground the answer on the closest available evidence, state explicitly what was used, and proceed with the computation rather than refusing.\n"
            "8. If a numeric operand is genuinely absent, substitute a clearly labeled assumption (e.g., 'assuming FY2022 capex ~ FY2021 disclosed value, $1.6B') and continue - never output 'Insufficient evidence.' as a final answer.\n"
            "\n"
            f"CONTEXT:\n{context}\n"
            "\n"
            f"QUESTION: {user_query}\n"
        )

    async def _run_non_agentic_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        _ = history
        retrieval_query = self._strip_format_instruction(user_query)
        context, nodes = await self.grag.retrieve(retrieval_query, top_k=RAGConfig.DEFAULT_TOP_K)
        retrieved_nodes = nodes if isinstance(nodes, list) else []
        unique_sources = self._build_unique_sources(retrieved_nodes)

        trace: list[dict[str, Any]] = [
            {
                "step": "agentic_off_retrieve",
                "input": {"query": user_query, "top_k": RAGConfig.DEFAULT_TOP_K},
                "output": {"retrieved_sources": len(unique_sources)},
            }
        ]

        if not context:
            answer = self._ensure_answer_prefix("Insufficient evidence.")
            trace.append(
                {
                    "step": "agentic_off_answer",
                    "input": {"query": user_query},
                    "output": {"answer": answer, "reason": "empty_context"},
                }
            )
            return answer, unique_sources, trace

        prompt = self._build_simple_answer_prompt(context, user_query)
        messages = [{"role": "user", "content": prompt}]
        answer = await self.llm.generate_response(messages)
        answer = self._ensure_answer_prefix(str(answer or ""))
        trace.append(
            {
                "step": "agentic_off_answer",
                "input": {"query": user_query},
                "output": {"answer": answer},
            }
        )
        return answer, unique_sources, trace

    # ---------- main entry ----------
    async def run_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        agentic_mode = str(os.environ.get("RAG_AGENTIC_MODE", "") or "").strip().lower()
        if agentic_mode == "off":
            logger.info("HypoReflect running in agentic off mode.")
            return await self._run_non_agentic_workflow(user_query, history)

        state = AgentState(user_query, history or [])
        await self.perception.run(state)
        await self.planning.run(state)
        await self.execution.run(state)

        if RAGConfig.ENABLE_AGENT_REFLECTION:
            reflection_passed = await self.reflection.run(state)
            await self.refinement_loop.run_loop(state, reflection_passed)
        else:
            logger.info("Ablation: Skipping Reflection & Refinement stages.")

        state.final_answer = self._ensure_answer_prefix(state.final_answer)

        unique_sources = self._build_unique_sources(state.all_context_data)
        return state.final_answer, unique_sources, state.trace
