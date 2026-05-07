"""Adaptive Context-Aware Chunking with rolling context (paper §3.1.2).

Two-level hierarchy:
- Level 1 — page-level grouping by cosine similarity over page-summary embeddings
  (threshold tau_page = 0.5 via RAGConfig.PAGE_SIMILARITY_THRESHOLD).
- Level 2 — sentence-level adaptive splitting within each page cluster
  (threshold tau_chunk = 0.65 via RAGConfig.SIMILARITY_THRESHOLD,
   minimum sentences per chunk M_min = 2 via RAGConfig.MIN_CHUNK_SENTENCES).

Each chunk is enriched with rolling context [anchor; milestone; prev-summary]
before Q-/Q+ generation. The non-OCR table-to-text fallback also lives here
because it operates inside the sentence iteration of Level 2.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

import numpy as np

from core.config import RAGConfig
from utils.prompts import (
    GROUP_SUMMARY_PROMPT,
    PAGE_SUMMARY_PROMPT,
    TABLE_TO_TEXT_PROMPT,
)


logger = logging.getLogger(__name__)


def _cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def _make_semantic_chunk_id(source, title, sent_id):
    content_sig = f"{source}-{title}-{sent_id}"
    return hashlib.md5(content_sig.encode()).hexdigest()


class ChunkingMixin:
    def _save_debug(self, doc_name: str, step: str, data: Any):
        doc_dir = os.path.join(self.debug_output_dir, doc_name.replace(" ", "_").replace("/", "_"))
        os.makedirs(doc_dir, exist_ok=True)
        filepath = os.path.join(doc_dir, f"{step}.json")
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        logger.info("[DEBUG] Saved %s to %s", step, filepath)

    async def _table_to_text(self, table_lines: list[str]) -> list[str]:
        """Sentence-by-sentence rendering of a markdown-pipe table.

        Used inside Level-2 sentence iteration when OCR'd input still contains
        raw `|`-delimited table fragments. The OCR pipeline (§3.1.1) is the
        primary table-to-text path; this is a fallback.
        """
        if not table_lines:
            return []

        if not RAGConfig.ABLATION_TABLE_TO_TEXT:
            logger.info("Ablation: Skipping table-to-text conversion.")
            return table_lines

        table_text = "\n".join(table_lines)
        prompt = TABLE_TO_TEXT_PROMPT + f"\nTABLE:\n{table_text}"
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.llm.generate_response(messages, apply_default_sampling=False)
            converted = [sentence.strip() for sentence in response.split("\n") if sentence.strip()]
            if not converted:
                raise ValueError("Empty conversion result")
            return converted
        except Exception as error:
            logger.warning("Table conversion failed (%s), using structured fallback", error)
            fallback: list[str] = []
            headers: list[str] = []
            for line in table_lines:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                if not headers:
                    headers = cells
                else:
                    pairs = [f"{header}: {value}" for header, value in zip(headers, cells) if value and value != "-"]
                    if pairs:
                        fallback.append(", ".join(pairs) + ".")
            return fallback if fallback else table_lines

    async def extract_knowledge(self, content: str, source: str = "") -> dict[str, Any]:
        lines = content.split("\n")
        title = "Unknown"
        if lines and lines[0].startswith("Document: "):
            title = lines[0].replace("Document: ", "").strip()
        elif lines and lines[0].startswith("Title: "):
            title = lines[0].replace("Title: ", "").strip()

        logger.info("[%s] Content head (500 chars): %r", title, content[:500])

        page_pattern = re.compile(r"-+\s*Page\s*(\d+)\s*-+", re.IGNORECASE)
        matches = list(page_pattern.finditer(content))

        pages = []
        if matches:
            for index, start_match in enumerate(matches):
                page_num = int(start_match.group(1))
                content_start = start_match.end()
                if index < len(matches) - 1:
                    content_end = matches[index + 1].start()
                else:
                    content_end = len(content)

                page_text = content[content_start:content_end].strip()
                if page_text:
                    pages.append({"num": page_num, "content": page_text})

        logger.info("[%s] Parsed %d pages from content.", title, len(pages))

        if not pages:
            logger.info("[%s] No page markers found, using standard chunking fallback", title)
            lines = content.split("\n")
            start_idx = 0
            if lines and (lines[0].startswith("Title: ") or lines[0].startswith("Document: ")):
                start_idx = 1
            content_body = "\n".join(lines[start_idx:])
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content_body) if s.strip()]
            pages = [{"num": 1, "content": " ".join(sentences)}]
            if not sentences:
                return {"title": title, "chunks": []}

        # ----- Level 1: page-level grouping (paper §3.1.2) -----
        if not RAGConfig.ABLATION_ADAPTIVE_CHUNKING:
            logger.info("Ablation: Using fixed page-based grouping (no adaptive similarity).")
            page_groups = []
            page_summaries = [page["content"][:200] for page in pages]
            for index, page in enumerate(pages):
                page_groups.append({
                    "pages": [page["num"]],
                    "content": page["content"],
                    "start_page": page["num"],
                    "page_summaries": [page_summaries[index]],
                    "group_summary": "",
                })
        else:
            async def get_page_summary(page_text: str) -> str:
                prompt = PAGE_SUMMARY_PROMPT.format(text=page_text[:2000])
                messages = [{"role": "user", "content": prompt}]
                try:
                    return await self.indexing_llm.generate_response(messages, apply_default_sampling=False)
                except Exception:
                    sentences = re.split(r"(?<=[.!?])\s+", page_text.strip())
                    return " ".join(sentences[:2]) if sentences else page_text[:200]

            page_summary_tasks = [get_page_summary(page["content"]) for page in pages]
            page_summaries = await asyncio.gather(*page_summary_tasks)
            logger.info("[%s] Generated %d page summaries via LLM", title, len(page_summaries))

            page_embeds = await self.llm.get_embeddings(page_summaries)
            page_groups = []
            current_group_start = 0
            page_similarity_threshold = RAGConfig.PAGE_SIMILARITY_THRESHOLD

            for index in range(len(pages)):
                should_split = False
                if index < len(pages) - 1:
                    similarity = _cosine_similarity(page_embeds[index], page_embeds[index + 1])
                    if similarity < page_similarity_threshold:
                        should_split = True
                else:
                    should_split = True

                if should_split:
                    group_pages = pages[current_group_start:index + 1]
                    group_content = "\n\n".join([page["content"] for page in group_pages])
                    group_page_range = [page["num"] for page in group_pages]
                    group_page_summaries = [page_summaries[j] for j in range(current_group_start, index + 1)]
                    page_groups.append({
                        "pages": group_page_range,
                        "content": group_content,
                        "start_page": group_page_range[0],
                        "page_summaries": group_page_summaries,
                    })
                    current_group_start = index + 1

            logger.info("[%s] Grouped %d pages into %d semantic groups", title, len(pages), len(page_groups))

            async def summarize_group(group):
                page_summaries_text = "\n".join([f"- {summary}" for summary in group["page_summaries"]])
                prompt = GROUP_SUMMARY_PROMPT.format(page_summaries=page_summaries_text)
                messages = [{"role": "user", "content": prompt}]
                return await self.indexing_llm.generate_response(messages, apply_default_sampling=False)

            group_summary_tasks = [summarize_group(group) for group in page_groups]
            group_summaries = await asyncio.gather(*group_summary_tasks)

            for index, group in enumerate(page_groups):
                group["group_summary"] = group_summaries[index].strip() if group_summaries[index] else ""

            logger.info("[%s] Generated %d group summaries", title, len(group_summaries))

        self._save_debug(title, "step1_page_summaries", [
            {"page": page["num"], "summary": summary}
            for page, summary in zip(pages, page_summaries)
        ])
        self._save_debug(title, "step2_page_groups", [
            {
                "group_idx": index,
                "pages": group["pages"],
                "start_page": group["start_page"],
                "page_summaries": group["page_summaries"],
                "group_summary": group["group_summary"],
            }
            for index, group in enumerate(page_groups)
        ])

        # ----- Level 2: sentence-level adaptive splitting + rolling context -----
        final_chunks = []
        global_sent_id = 0
        first_group_summary = page_groups[0].get("group_summary", "") if page_groups else ""
        intro_summary = first_group_summary if first_group_summary else f"Document: {title}"
        milestone_summaries = []
        recent_summary = ""
        milestone_interval = RAGConfig.MILESTONE_INTERVAL
        chunk_sem = asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_LLM_CALLS)

        async def process_group(pg_idx, page_group):
            nonlocal global_sent_id, recent_summary
            group_content = page_group["content"]
            group_start_page = page_group["start_page"]
            if not group_content:
                return []

            raw_sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", group_content) if sentence.strip()]
            if not raw_sentences:
                return []

            processed_sentences = []
            table_buffer = []
            for line in raw_sentences:
                if "|" in line:
                    table_buffer.append(line)
                else:
                    if table_buffer:
                        processed_sentences.extend(await self._table_to_text(table_buffer))
                        table_buffer = []
                    processed_sentences.append(line)
            if table_buffer:
                processed_sentences.extend(await self._table_to_text(table_buffer))

            group_chunks = []
            sent_embeds = await self.llm.get_embeddings(processed_sentences)

            if len(sent_embeds) > 1:
                similarities = [
                    _cosine_similarity(sent_embeds[k], sent_embeds[k + 1])
                    for k in range(len(sent_embeds) - 1)
                ]
                avg_sim = sum(similarities) / len(similarities)
                adaptive_threshold = min(RAGConfig.SIMILARITY_THRESHOLD, avg_sim - 0.1)
            else:
                adaptive_threshold = RAGConfig.SIMILARITY_THRESHOLD

            min_chunk_sentences = RAGConfig.MIN_CHUNK_SENTENCES
            current_group = []

            for index in range(len(processed_sentences)):
                current_group.append(processed_sentences[index])
                should_split = False
                if index < len(processed_sentences) - 1:
                    if len(current_group) >= min_chunk_sentences:
                        similarity = _cosine_similarity(sent_embeds[index], sent_embeds[index + 1])
                        if similarity < adaptive_threshold:
                            should_split = True
                else:
                    should_split = True

                if should_split:
                    chunk_text = " ".join(current_group)

                    async with chunk_sem:
                        if not RAGConfig.ABLATION_ROLLING_SUMMARY:
                            rolling_context = f"Document: {title}"
                        else:
                            context_parts = [intro_summary]
                            if milestone_summaries:
                                context_parts.append(f"Key points: {' | '.join(milestone_summaries[-2:])}")
                            if recent_summary:
                                context_parts.append(f"Recent: {recent_summary}")
                            rolling_context = " || ".join(context_parts)

                        q_data = await self.extract_hoprag_queries_with_rolling(chunk_text, title, rolling_context)

                    group_chunks.append({
                        "page": group_start_page,
                        "text": chunk_text,
                        "title": title,
                        "q_minus": q_data.get("q_minus", []),
                        "q_plus": q_data.get("q_plus", []),
                        "summary": q_data.get("summary", ""),
                    })

                    new_summary = q_data.get("summary", "")
                    if new_summary:
                        recent_summary = new_summary

                    current_group = []

            return group_chunks

        for index, page_group in enumerate(page_groups):
            logger.info(
                "[%s] Processing Page Group %d/%d (Pages: %s)",
                title,
                index + 1,
                len(page_groups),
                page_group["pages"],
            )
            group_chunks = await process_group(index, page_group)
            for group_chunk in group_chunks:
                group_chunk["sent_id"] = global_sent_id
                final_chunks.append(group_chunk)
                global_sent_id += 1

                if global_sent_id % milestone_interval == 0 and group_chunk["summary"]:
                    milestone_summaries.append(group_chunk["summary"])

        self._save_debug(title, "step3_final_chunks", [
            {
                "sent_id": chunk["sent_id"],
                "page": chunk["page"],
                "text": chunk["text"][:200] + "...",
                "q_minus": chunk["q_minus"],
                "q_plus": chunk["q_plus"],
                "summary": chunk["summary"],
            }
            for chunk in final_chunks
        ])

        return {"title": title, "chunks": final_chunks}
