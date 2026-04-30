from pydantic import BaseModel, Field
from typing import List, Dict, Any

class GraphSearchArgs(BaseModel):
    """Arguments for querying the knowledge graph (Neo4j)."""
    entities: List[str] = Field(..., description="List of key entities found in the question to search for in the graph.")
    depth: int = Field(default=2, description="Traversal depth for finding relationships (max 4; 1=direct, higher=deeper multi-hop).")
    top_k: int = Field(default=5, description="Maximum number of passages to return.")


class RetrieveArgs(BaseModel):
    """Arguments for hybrid retrieval fallback/search."""
    query: str = Field(..., description="Natural language query for retrieval.")
    top_k: int = Field(default=5, description="Maximum number of passages to return.")


class CalculatorArgs(BaseModel):
    """Arguments for deterministic arithmetic calculation."""
    expression: str = Field(
        ...,
        description="Arithmetic expression using numbers and operators (+,-,*,/,**,( )).",
    )
    precision: int | None = Field(
        default=None,
        description="Optional decimal places for rounding final result (0-8).",
    )


def get_graph_search_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Use this when entity relationship expansion is needed (multi-hop over graph links).",
            "parameters": GraphSearchArgs.model_json_schema()
        }
    }


def get_retrieve_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Use this for general evidence gathering via hybrid vector+text retrieval.",
            "parameters": RetrieveArgs.model_json_schema()
        }
    }


def get_calculator_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Use this for deterministic arithmetic once compute operands are grounded. "
                "Do not use for retrieval."
            ),
            "parameters": CalculatorArgs.model_json_schema(),
        },
    }


def get_all_tools() -> List[Dict[str, Any]]:
    return [
        get_graph_search_tool_schema(),
        get_calculator_tool_schema(),
    ]
