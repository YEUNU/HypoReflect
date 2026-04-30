from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatContext(BaseModel):
    text: str
    source: str
    score: float
    page: int = 1

class PageRegion(BaseModel):
    type: str
    bbox: List[int]
    text: str

class PageInfo(BaseModel):
    page: int
    image_url: Optional[str] = None
    width: int
    height: int
    regions: List[PageRegion]
    page_text: str
    summary: Optional[str] = None

class GraphNodeSample(BaseModel):
    id: str
    source: str | None = None

class GraphEdgeSample(BaseModel):
    src: str
    dst: str
    type: str
    score: float | None = None
