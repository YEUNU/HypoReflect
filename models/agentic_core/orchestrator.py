from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core.config import RAGConfig
from models.agentic_core.prompts import (
    AGENTIC_PLAN_FORMAT_INSTRUCTION,
    AGENTIC_PLAN_PROMPT,
    AGENTIC_REFLECTION_FORMAT_INSTRUCTION,
    AGENTIC_REFLECTION_PROMPT,
    AGENTIC_SYNTHESIS_PROMPT,
)


class RetrievalBackend(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        ...


class AgenticOrchestrator:
    """Optional model-agnostic agentic loop: plan -> retrieve -> synthesize -> reflect."""

    def __init__(
        self,
        llm: Any,
        backend: RetrievalBackend,
        strategy_name: str,
        top_k: Optional[int] = None,
        max_plan_queries: int = 3,
        enable_reflection: bool = True,
    ) -> None:
        self.llm = llm
        self.backend = backend
        self.strategy_name = strategy_name
        self.top_k = max(1, int(top_k if top_k is not None else RAGConfig.DEFAULT_TOP_K))
        self.max_plan_queries = max(1, int(max_plan_queries))
        self.enable_reflection = bool(enable_reflection)
        self.logger = logging.getLogger(__name__)

    async def run(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        _ = history
        trace: List[Dict[str, Any]] = []

        planned_queries = await self._plan_queries(query)
        trace.append(
            {
                "step": "agentic_plan",
                "output": {
                    "strategy": self.strategy_name,
                    "planned_queries": planned_queries,
                },
            }
        )

        retrieved_nodes = await self._retrieve_all(planned_queries, trace)
        if not retrieved_nodes:
            answer = "@@ANSWER: insufficient evidence\n@@EVIDENCE:\n- none"
            trace.append(
                {
                    "step": "agentic_synthesis",
                    "output": {
                        "answer": answer,
                        "reason": "no_retrieved_nodes",
                    },
                }
            )
            return answer, [], trace

        packed_nodes = self._dedupe_nodes(retrieved_nodes)
        context = self._render_context(packed_nodes)
        answer = await self._synthesize(query, context)
        trace.append(
            {
                "step": "agentic_synthesis",
                "output": {
                    "answer": answer,
                    "candidate_count": len(packed_nodes),
                },
            }
        )

        if self.enable_reflection:
            reflected = await self._reflect(query, context, answer)
            trace.append(
                {
                    "step": "agentic_reflection",
                    "output": reflected,
                }
            )
            if reflected.get("verdict") == "FAIL":
                revised = str(reflected.get("revised_answer", "") or "").strip()
                if revised:
                    answer = revised

        sources = [self._to_source(n) for n in packed_nodes[: self.top_k]]
        return answer, sources, trace

    async def _plan_queries(self, query: str) -> List[str]:
        plan_prompt = AGENTIC_PLAN_PROMPT
        messages = [
            {"role": "user", "content": plan_prompt.format(query=query)},
            {"role": "user", "content": AGENTIC_PLAN_FORMAT_INSTRUCTION},
        ]
        payload = await self.llm.generate_json(messages, temperature=0.0)
        raw_queries = payload.get("queries", []) if isinstance(payload, dict) else []

        planned: List[str] = []
        if isinstance(raw_queries, list):
            for candidate in raw_queries:
                text = str(candidate or "").strip()
                if not text:
                    continue
                planned.append(text)

        if query not in planned:
            planned.insert(0, query)

        deduped: List[str] = []
        seen = set()
        for q in planned:
            key = q.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)

        return deduped[: self.max_plan_queries]

    async def _retrieve_all(self, queries: List[str], trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_nodes: List[Dict[str, Any]] = []
        for q in queries:
            try:
                _, nodes = await self.backend.retrieve(q, top_k=self.top_k)
                count = len(nodes) if isinstance(nodes, list) else 0
                trace.append(
                    {
                        "step": "agentic_retrieve",
                        "input": {"query": q, "top_k": self.top_k},
                        "output": {"nodes": count},
                    }
                )
                if isinstance(nodes, list):
                    all_nodes.extend(nodes)
            except Exception as e:
                self.logger.warning("Agentic retrieval failed for query '%s': %s", q, e)
                trace.append(
                    {
                        "step": "agentic_retrieve",
                        "input": {"query": q, "top_k": self.top_k},
                        "output": {"nodes": 0, "error": str(e)},
                    }
                )
        return all_nodes

    @staticmethod
    def _dedupe_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            key = (
                str(node.get("title") or node.get("doc") or "").strip(),
                int(node.get("page") or 0),
                int(node.get("sent_id") or node.get("chunk") or 0),
                str(node.get("text") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        return deduped

    @staticmethod
    def _render_context(nodes: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for node in nodes:
            doc = str(node.get("title") or node.get("doc") or "Unknown")
            page = int(node.get("page") or 0)
            chunk = int(node.get("sent_id") or node.get("chunk") or 0)
            text = str(node.get("text") or "").strip()
            blocks.append(f"[[{doc}, Page {page}, Chunk {chunk}]]\n{text}")
        return "\n\n---\n\n".join(blocks)

    async def _synthesize(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "user",
                "content": AGENTIC_SYNTHESIS_PROMPT.format(query=query, context=context),
            }
        ]
        return await self.llm.generate_response(messages, temperature=0.0)

    async def _reflect(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": AGENTIC_REFLECTION_PROMPT.format(
                    query=query,
                    answer=answer,
                    context=context,
                ),
            },
            {"role": "user", "content": AGENTIC_REFLECTION_FORMAT_INSTRUCTION},
        ]
        payload = await self.llm.generate_json(messages, temperature=0.0)
        if not isinstance(payload, dict):
            return {"verdict": "PASS", "issues": [], "revised_answer": ""}
        verdict = str(payload.get("verdict", "PASS") or "PASS").strip().upper()
        payload["verdict"] = "PASS" if verdict not in {"PASS", "FAIL"} else verdict
        if "issues" not in payload or not isinstance(payload.get("issues"), list):
            payload["issues"] = []
        return payload

    @staticmethod
    def _to_source(node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "doc": str(node.get("title") or node.get("doc") or ""),
            "page": int(node.get("page") or 0),
            "text": str(node.get("text") or ""),
            "sent_id": int(node.get("sent_id") or node.get("chunk") or 0),
        }
