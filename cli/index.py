import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from core.config import RAGConfig
from models.hyporeflect.graphrag import GraphRAG
from models.naive.naive_rag import NaiveRAG


logger = logging.getLogger("HypoReflect")


async def run_indexing(
    dataset_path: str,
    strategy: str,
    model_id: str,
    corpus_tag: Optional[str] = None,
    save_intermediate: bool = False,
    sample_companies: Optional[list[str]] = None,
    save_to: Optional[str] = None,
):
    """Index files using selected strategy with parallel processing."""
    logger.info(
        "Indexing strategy: %s | Dataset: %s | Corpus: %s | Samples: %d",
        strategy,
        dataset_path,
        corpus_tag or "default",
        len(sample_companies) if sample_companies else 0,
    )

    if strategy in ["hyporeflect", "hoprag", "ms_graphrag"]:
        engine = GraphRAG(
            strategy=strategy,
            indexing_model_id=model_id,
            corpus_tag=corpus_tag,
            save_intermediate=save_intermediate,
        )
        is_graph = True
    elif strategy == "naive":
        engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
        is_graph = False
    else:
        logger.error("Unknown strategy: %s", strategy)
        return

    if not os.path.exists(dataset_path):
        logger.error("Path %s not found.", dataset_path)
        return

    files = sorted([file for file in os.listdir(dataset_path) if file.endswith((".txt", ".md"))])

    if sample_companies:
        doc_info_path = "data/financebench_document_information.jsonl"
        if os.path.exists(doc_info_path):
            with open(doc_info_path, "r", encoding="utf-8") as file:
                doc_data = [json.loads(line) for line in file]

            valid_docs = {item["doc_name"] for item in doc_data if item.get("company") in sample_companies}
            filtered_files = []
            for file_name in files:
                stem = Path(file_name).stem
                if stem in valid_docs:
                    filtered_files.append(file_name)
                else:
                    parts = stem.rsplit("_page_", 1)
                    if len(parts) == 2 and parts[0] in valid_docs:
                        filtered_files.append(file_name)

            logger.info(
                "Filtering for %d sample companies: %d -> %d files",
                len(sample_companies),
                len(files),
                len(filtered_files),
            )
            files = filtered_files

            if save_to:
                try:
                    save_dir = Path(save_to)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Saving %d sampled files to %s...", len(files), save_dir)
                    for file_name in files:
                        src_path = os.path.join(dataset_path, file_name)
                        dst_path = save_dir / file_name
                        shutil.copy2(src_path, dst_path)
                    logger.info("Successfully saved files to %s", save_dir)
                except Exception as exc:
                    logger.error("Error saving sampled files to %s: %s", save_to, exc)
        else:
            logger.warning(
                "Document info file not found at %s, cannot filter by sample companies.",
                doc_info_path,
            )

    semaphore = asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_LLM_CALLS)
    progress = {"count": 0, "lock": asyncio.Lock()}
    processed_docs = []
    failed_files = []
    stats = {"succeeded": 0}

    async def process_file(filename: str, content: str):
        async with semaphore:
            async with progress["lock"]:
                progress["count"] += 1
                if progress["count"] % 10 == 0:
                    logger.info("Indexing progress: %d/%d", progress["count"], len(files))

            try:
                if is_graph:
                    knowledge = await engine.extract_knowledge(content)
                    doc_id = await engine.create_document_node(filename, {"title": knowledge["title"]})
                    await engine.build_graph(knowledge, source=filename, document_filename=doc_id)
                    async with progress["lock"]:
                        processed_docs.append(doc_id)
                else:
                    await engine.index_document(filename, content)
                async with progress["lock"]:
                    stats["succeeded"] += 1
            except Exception as exc:
                logger.error("Failed to index file %s: %s", filename, exc)
                async with progress["lock"]:
                    failed_files.append((filename, str(exc)))

    file_contents = []
    for filename in files:
        try:
            with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as file:
                file_contents.append((filename, file.read()))
        except Exception as exc:
            logger.error("Failed to read file %s: %s", filename, exc)
            failed_files.append((filename, f"read_error: {exc}"))

    gather_results = await asyncio.gather(
        *[process_file(filename, content) for filename, content in file_contents],
        return_exceptions=True,
    )
    for idx, result in enumerate(gather_results):
        if isinstance(result, Exception):
            filename = file_contents[idx][0]
            logger.error("Unhandled indexing task error in %s: %s", filename, result)
            failed_files.append((filename, f"task_error: {result}"))

    if is_graph:
        await engine.flush_graph_batch()
        logger.info("Summarizing %d documents...", len(processed_docs))

        async def summarize_with_semaphore(doc_id):
            async with semaphore:
                try:
                    await engine.summarize_document(doc_id)
                except Exception as exc:
                    logger.error("Failed to summarize document %s: %s", doc_id, exc)
                    failed_files.append((doc_id, f"summarize_error: {exc}"))

        await asyncio.gather(*[summarize_with_semaphore(doc_id) for doc_id in processed_docs])

    logger.info(
        "Indexing complete for %d files. Success: %d | Failed: %d",
        len(files),
        stats["succeeded"],
        len(failed_files),
    )
    if failed_files:
        preview = ", ".join([f"{name}" for name, _ in failed_files[:10]])
        logger.warning("Indexing failures (up to 10): %s", preview)
