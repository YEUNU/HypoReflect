import json
import re
import os
from typing import Any, Optional, Dict

def clean_and_parse_json(text: str) -> Optional[Any]:
    """
    Strict JSON parser (no markdown/unwrapping heuristics).
    """
    if not text or not isinstance(text, str):
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def clean_and_unwrap_json(text: str) -> str:
    """
    If the text is a JSON object with a 'response', 'content', 'answer', or 'result' key,
    extracts and returns that value.
    """
    if not text or not isinstance(text, str):
        return text
    
    current_text = text.strip()
    
    # Heuristic: If it doesn't look like JSON (no braces), skip immediately
    if '{' not in current_text:
        return text

    max_depth = 3 # Safety limit for accidental infinite recursion
    for _ in range(max_depth):
        parsed = clean_and_parse_json(current_text)
        
        if isinstance(parsed, dict):
            found = False
            # Check for common response keys
            for key in ["response", "content", "answer", "result", "output", "message"]:
                if key in parsed and isinstance(parsed[key], str):
                    current_text = parsed[key].strip()
                    found = True
                    break
            
            if not found:
                break
        else:
            break
            
    return current_text

def resolve_metadata_with_fallback(g_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Resolves metadata with fallback strategy.
    """
    if not g_data or not isinstance(g_data, dict):
        g_data = {}

    title = g_data.get("title") or os.path.splitext(filename)[0]
    category = g_data.get("category") or "기타"
    summary = g_data.get("summary") or "Summary unavailable"
    keywords = g_data.get("keywords") or []

    meta = g_data.get("metadata", {})
    project_name = meta.get("project_name")
    year_raw = meta.get("year")
    author = meta.get("author")
    status_raw = meta.get("status")

    year = None
    if year_raw:
        try:
            year = int(str(year_raw))
        except ValueError:
            pass
            
    if not year:
        ym = re.search(r'(20\d{2})', filename)
        if ym: 
            year = int(ym.group(1))

    status = "Unknown"
    if status_raw:
        if status_raw.lower() in ["draft", "초안"]: status = "Draft"
        elif status_raw.lower() in ["final", "확정", "최종"]: status = "Final"
        elif status_raw in ["Draft", "Final", "Unknown"]: status = status_raw
    
    if status == "Unknown":
        lower_name = filename.lower()
        if any(x in lower_name for x in ['draft', '초안', 'v0.']):
            status = "Draft"
        elif any(x in lower_name for x in ['final', '확정', '최종']):
            status = "Final"

    if project_name in ["N/A", "Unknown", None]: project_name = None
    if author in ["N/A", "Unknown", None]: author = None

    return {
        "title": title,
        "category": category,
        "summary": summary,
        "keywords": keywords,
        "project_name": project_name,
        "year": year,
        "author": author,
        "status": status
    }
