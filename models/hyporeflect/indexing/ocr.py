"""Topology-Preserving OCR runner (paper §3.1.1).

VLM OCR step that renders financial tables as Markdown to preserve
row-column topology, preventing value-context dissociation that occurs
with plain-text flattening. Outputs both raw page metadata (.json) and
concatenated text (.txt) under output_dir/{raw,text}.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from core.config import RAGConfig
from utils.ocr_tools import process_pdf_file


logger = logging.getLogger("HypoReflect")


async def run_ocr(
    pdf_dir: str,
    output_dir: str,
    convert_tables: bool = True,
    sample_companies: Optional[list[str]] = None,
):
    """Run OCR on PDF files and save extracted text and raw metadata."""
    pdf_path = Path(pdf_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_path = out_path / "raw"
    text_path = out_path / "text"
    raw_path.mkdir(parents=True, exist_ok=True)
    text_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        logger.error("PDF directory not found: %s", pdf_dir)
        return

    pdf_files = sorted([file for file in pdf_path.glob("*.pdf")])

    if sample_companies:
        valid_pdfs = []
        doc_info_path = "data/financebench_document_information.jsonl"
        valid_doc_names = set()

        if os.path.exists(doc_info_path):
            with open(doc_info_path, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        item = json.loads(line)
                        if item.get("company") in sample_companies:
                            valid_doc_names.add(item["doc_name"])
                    except Exception:
                        continue

        if valid_doc_names:
            for file in pdf_files:
                if file.stem in valid_doc_names:
                    valid_pdfs.append(file)
            logger.info(
                "OCR Sampling: Filtered %d -> %d PDFs for %d companies.",
                len(pdf_files),
                len(valid_pdfs),
                len(sample_companies),
            )
            pdf_files = valid_pdfs
        else:
            logger.warning(
                "Could not determine valid doc names from sample companies for OCR filtering. Processing all."
            )

    to_process = []
    for file in pdf_files:
        doc_name = file.stem
        if (text_path / f"{doc_name}.txt").exists() and (raw_path / f"{doc_name}.json").exists():
            continue
        to_process.append(file)

    skipped_count = len(pdf_files) - len(to_process)
    pdf_files = to_process

    logger.info("=== OCR Processing %d PDFs (Skipped: %d) ===", len(pdf_files), skipped_count)
    logger.info("Input: %s | Output: %s", pdf_dir, output_dir)

    async def progress_callback(current: int, total: int, message: str = None):
        if message:
            logger.info("  [%d/%d] %s", current, total, message)

    semaphore = asyncio.Semaphore(RAGConfig.MAX_PARALLEL_PDFS)

    async def process_with_semaphore(pdf_file: Path, index: int):
        async with semaphore:
            logger.info("\n[%d/%d] Starting: %s", index + 1, len(pdf_files), pdf_file.name)
            try:
                full_text, total_pages, pages_meta, _ = await process_pdf_file(
                    str(pdf_file),
                    progress_callback,
                    convert_tables=convert_tables,
                )

                doc_name = pdf_file.stem
                output_file = text_path / f"{doc_name}.txt"
                header = f"Document: {doc_name}\n"
                header += f"Pages: {total_pages}\n"
                header += "\n" + "=" * 50 + "\n\n"
                output_file.write_text(header + full_text, encoding="utf-8")
                logger.info("  Finished: %s -> %s", pdf_file.name, output_file.name)

                raw_data = {
                    "doc_name": doc_name,
                    "total_pages": total_pages,
                    "pages": pages_meta,
                }
                raw_file = raw_path / f"{doc_name}.json"
                with open(raw_file, "w", encoding="utf-8") as file:
                    json.dump(raw_data, file, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.error("  Failed to process %s: %s", pdf_file.name, exc)

    await asyncio.gather(*[process_with_semaphore(file, idx) for idx, file in enumerate(pdf_files)])
    logger.info("\n=== OCR Complete ===")
    logger.info("Output directory: %s", out_path.absolute())
