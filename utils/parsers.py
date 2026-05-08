import json
from typing import Any, Optional

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

