"""
OCR Tools - PDF Processing Pipeline for HypoReflect

Migrated from i-bp project with following changes:
- Removed Celery dependency (pure asyncio)
- Removed Django ORM dependency (file-based)
- Simplified progress callback
"""
import os
import base64
import io
import unicodedata
import asyncio
import logging
from typing import Optional, Callable, List, Dict, Any, Tuple

from PIL import Image
from pdf2image import convert_from_bytes, pdfinfo_from_bytes

from core.vllm_client import VLLMClient
from core.config import RAGConfig
from utils.prompts import TABLE_TO_TEXT_PROMPT
import re

logger = logging.getLogger(__name__)

# Global VLLMClient instance
vllm = VLLMClient()


def _detect_markdown_tables(text: str) -> List[str]:
    """Detect Markdown tables in text and return list of table strings."""
    # Pattern: lines starting with | and containing |
    lines = text.split('\n')
    tables = []
    current_table = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and '|' in stripped[1:]:
            current_table.append(line)
        else:
            if len(current_table) >= 2:  # At least header + separator or data row
                tables.append('\n'.join(current_table))
            current_table = []
    
    # Don't forget the last table
    if len(current_table) >= 2:
        tables.append('\n'.join(current_table))
    
    return tables


async def _convert_table_to_text(table: str) -> str:
    """Convert a single Markdown table to natural language sentences using LLM."""
    try:
        prompt = TABLE_TO_TEXT_PROMPT + f"\n\nTABLE:\n{table}"
        messages = [{"role": "user", "content": prompt}]
        result = await vllm.generate_response(messages)
        return result.strip()
    except Exception as e:
        logger.warning(f"Table conversion failed: {e}")
        return table  # Return original table if conversion fails


async def convert_tables_in_text(text: str) -> str:
    """Find all tables in text and convert them to natural language sentences in parallel."""
    tables = _detect_markdown_tables(text)
    
    if not tables:
        return text
    
    # Process all tables in parallel
    tasks = [asyncio.create_task(_convert_table_to_text(t)) for t in tables]
    converted_list = await asyncio.gather(*tasks)
    
    result = text
    for original, converted in zip(tables, converted_list):
        result = result.replace(original, converted)
    
    logger.info(f"  Converted {len(tables)} tables to natural language (Parallel)")
    return result

async def _process_single_page(
    img: Image.Image, 
    page_num: int, 
    filename: str, 
    progress_callback: Optional[Callable] = None,
    pages_meta: Optional[List[Dict]] = None,
    total_pages: int = 0,
    convert_tables: bool = True
) -> Tuple[Dict, str]:
    """
    Single Page OCR Processor.
    
    Runs Vision Language Model on a single page image.
    
    Args:
        img: PIL Image of the page
        page_num: 0-indexed page number
        filename: Source PDF filename
        progress_callback: Optional async callback for progress updates
        pages_meta: List to append page metadata to
        total_pages: Total number of pages for progress calculation
        
    Returns:
        Tuple of (page_metadata_dict, page_text_content)
    """
    page_meta = {
        'page': page_num + 1,
        'width': img.width,
        'height': img.height,
        'page_text': ''
    }
    
    if pages_meta is not None:
        if not any(p['page'] == page_num + 1 for p in pages_meta):
            pages_meta.append(page_meta)
        else:
            for p in pages_meta:
                if p['page'] == page_num + 1:
                    p.update(page_meta)
                    page_meta = p
                    break

    # VLM OCR
    logger.info(f"Page {page_num+1}: Calling OCR VLM ({vllm.ocr_model_name})...")
    try:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        if progress_callback:
            await progress_callback(page_num, total_pages * 2, f"Starting OCR Page {page_num+1}")
        
        # Empty prompt for pure OCR
        page_content = await vllm.generate_with_image(img_str, "")
        page_meta['page_text'] = page_content
        
        if progress_callback:
            await progress_callback(page_num, total_pages * 2, f"OCR Completed Page {page_num+1}")
        
        logger.info(f"  VLM extraction length: {len(page_content)} characters.")

        # Convert tables to natural language (for better RAG retrieval)
        if convert_tables:
            page_content = await convert_tables_in_text(page_content)
            page_meta['page_text'] = page_content

        return page_meta, page_content
    except Exception as e:
        logger.error(f"  VLM OCR failed for Page {page_num+1}: {e}")
        page_meta['page_text'] = f"[OCR Failed for Page {page_num+1}]"
        return page_meta, page_meta['page_text']


async def process_pdf_upload(
    file_bytes: bytes, 
    filename: str, 
    progress_callback: Optional[Callable] = None,
    convert_tables: bool = True
) -> Tuple[str, int, List[Dict], Dict]:
    """
    Orchestrator for the entire PDF-to-Text pipeline.
    
    Args:
        file_bytes: Raw PDF file bytes
        filename: Original filename
        progress_callback: Optional async callback(current, total, message)
        
    Returns:
        Tuple of (full_text, total_pages, pages_meta, global_context)
    """
    filename = unicodedata.normalize('NFC', filename)
    
    # Track maximum progress
    max_progress_val = 0
    
    async def safe_progress(current: int, total: int, message: Optional[str] = None):
        nonlocal max_progress_val
        if progress_callback:
            if current > max_progress_val:
                max_progress_val = current
            await progress_callback(max_progress_val, total, message)

    try:
        # 0. Batch-based PDF Conversion to prevent OOM
        info = pdfinfo_from_bytes(file_bytes)
        total_pages = info["Pages"]
        
        images = []
        BATCH_SIZE = RAGConfig.PDF_BATCH_SIZE
        max_dim = RAGConfig.PDF_MAX_DIM
        dpi = RAGConfig.PDF_DPI
        thread_count = RAGConfig.PDF_CONVERT_THREADS
        
        for start_p in range(1, total_pages + 1, BATCH_SIZE):
            end_p = min(start_p + BATCH_SIZE - 1, total_pages)
            logger.info(f"Converting PDF batches: {start_p}-{end_p} / {total_pages}")
            
            batch_images = convert_from_bytes(
                file_bytes, 
                dpi=dpi, 
                first_page=start_p, 
                last_page=end_p,
                thread_count=thread_count
            )
            
            # Immediate Resize
            for img in batch_images:
                w, h = img.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                images.append(img)
        
        pages_meta = []
        sem = asyncio.Semaphore(RAGConfig.MAX_PARALLEL_PAGES)

        async def process_page_wrapper(i: int, img: Image.Image):
            async with sem:
                return await _process_single_page(img, i, filename, safe_progress, pages_meta, total_pages, convert_tables)

        # 1. Parallel OCR Phase
        tasks = [asyncio.create_task(process_page_wrapper(i, img)) for i, img in enumerate(images)]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            for t in tasks:
                if not t.done(): 
                    t.cancel()
            raise e

        pages_meta.sort(key=lambda x: x['page'])
        full_text_list = [f"--- Page {p['page']} ---\n{p['page_text']}" for p in pages_meta if p.get('page_text')]
        total_text = "\n\n".join(full_text_list)

        # 2. Global context output schema (summary stage removed)
        global_context = {"title": filename, "summary": "", "keywords": []}

        if progress_callback:
            await progress_callback(total_pages * 2, total_pages * 2, "Processing Completed")
            
        return total_text, total_pages, pages_meta, global_context

    except Exception as e:
        logger.error(f"Error in process_pdf_upload: {e}")
        raise e


async def process_pdf_file(
    filepath: str, 
    progress_callback: Optional[Callable] = None,
    convert_tables: bool = True
) -> Tuple[str, int, List[Dict], Dict]:
    """
    Convenience wrapper to process a PDF file from disk.
    
    Args:
        filepath: Path to the PDF file
        progress_callback: Optional async callback(current, total, message)
        convert_tables: If True, convert Markdown tables to natural language sentences
        
    Returns:
        Same as process_pdf_upload
    """
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    filename = os.path.basename(filepath)
    return await process_pdf_upload(file_bytes, filename, progress_callback, convert_tables)
