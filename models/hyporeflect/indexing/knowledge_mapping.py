"""Predictive Knowledge Mapping (paper §3.1.3).

Each chunk is annotated at indexing time with dual hypothetical queries:
- Q- (incoming): self-contained questions answerable from the chunk alone.
- Q+ (outgoing): questions the chunk only partially answers, pointing to its
  dependencies; later used as the ANN seed for HOP edge construction (§3.1.4).
"""
import logging
from typing import Any

from utils.prompts import HOPRAG_FORMAT_INSTRUCTION, HOPRAG_PROMPT


logger = logging.getLogger(__name__)


class KnowledgeMappingMixin:
    async def extract_hoprag_queries(self, chunk: str, title: str = "") -> dict[str, Any]:
        """Generate Q-/Q+ for a chunk without rolling context."""
        text_prompt = HOPRAG_PROMPT.format(chunk=chunk, global_context=f"Document Title: {title}")
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": HOPRAG_FORMAT_INSTRUCTION},
        ]
        try:
            data = await self.indexing_llm.generate_json(messages, apply_default_sampling=False)
            return {
                "q_minus": data.get("q_minus", []),
                "q_plus": data.get("q_plus", []),
                "summary": data.get("summary", ""),
            }
        except Exception:
            return {"q_minus": [], "q_plus": [], "summary": ""}

    async def extract_hoprag_queries_with_rolling(
        self,
        chunk: str,
        title: str,
        running_summary: str,
    ) -> dict[str, Any]:
        """Generate Q-/Q+ enriched with the rolling context (anchor + milestone +
        prev-summary) computed by the chunking stage (§3.1.2)."""
        text_prompt = HOPRAG_PROMPT.format(
            chunk=chunk,
            global_context=f"Document: {title}. Previous context: {running_summary}",
        )
        messages = [
            {"role": "user", "content": text_prompt},
            {"role": "user", "content": HOPRAG_FORMAT_INSTRUCTION},
        ]
        try:
            data = await self.indexing_llm.generate_json(messages, apply_default_sampling=False)
            return {
                "q_minus": data.get("q_minus", []),
                "q_plus": data.get("q_plus", []),
                "summary": data.get("summary", ""),
            }
        except Exception as error:
            logger.error("HopRAG extraction failed: %s", error)
            return {"q_minus": [], "q_plus": [], "summary": ""}
