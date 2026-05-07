"""HypoReflect service facade.

Thin entry point that wires the five stage handlers and delegates the
five-stage flow to :class:`Orchestrator`. Refinement ranking, R_max loop,
post-refinement overrides, and the non-agentic baseline live in
:mod:`models.hyporeflect.orchestrator` and :mod:`models.hyporeflect.stages.refinement`.
"""
from typing import Any, Optional

from core.config import RAGConfig
from core.vllm_client import get_llm_client
from models.hyporeflect.graphrag import GraphRAG
from models.hyporeflect.orchestrator import Orchestrator
from models.hyporeflect.stages.execution import ExecutionHandler
from models.hyporeflect.stages.perception import PerceptionHandler
from models.hyporeflect.stages.planning import PlanningHandler
from models.hyporeflect.stages.reflection import ReflectionHandler
from models.hyporeflect.stages.refinement import RefinementHandler


class AgentService:
    def __init__(
        self,
        model_id: str = "local",
        strategy: str = "hyporeflect",
        corpus_tag: str = "default",
        llm_override: Any = None,
        grag_override: Any = None,
    ):
        self.llm = llm_override if llm_override is not None else get_llm_client(model_id)
        self.grag = grag_override if grag_override is not None else GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
        self.perception = PerceptionHandler(self.llm, stage_model=RAGConfig.PERCEPTION_MODEL)
        self.planning = PlanningHandler(self.llm, stage_model=RAGConfig.PLANNING_MODEL)
        self.execution = ExecutionHandler(self.llm, self.grag, stage_model=RAGConfig.EXECUTION_MODEL)
        self.reflection = ReflectionHandler(self.llm, stage_model=RAGConfig.REFLECTION_MODEL)
        self.refinement = RefinementHandler(self.llm, stage_model=RAGConfig.REFINEMENT_MODEL)
        self._orchestrator = Orchestrator(
            llm=self.llm,
            grag=self.grag,
            perception=self.perception,
            planning=self.planning,
            execution=self.execution,
            reflection=self.reflection,
            refinement=self.refinement,
        )

    async def run_workflow(
        self,
        user_query: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple:
        return await self._orchestrator.run_workflow(user_query, history)
