from typing import List, Dict

def format_context_from_nodes(nodes: List[Dict]) -> str:
    """Standardized context formatting with inline citations [[Title, Page X, Chunk Y]]."""
    formatted = []
    for n in nodes:
        title = n.get('title', 'Unknown')
        page = n.get('page', 0)
        sent_id = n.get('sent_id', 0)
        text = n.get('text', '')
        formatted.append(f"[[{title}, Page {page}, Chunk {sent_id}]]\n{text}")
    return "\n\n".join(formatted)
