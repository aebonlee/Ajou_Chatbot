

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class QueryRequest(BaseModel):
    user_id: str
    channel_id: Optional[str] = None
    text: str

class Source(BaseModel):
    title: Optional[str] = None
    page: Optional[int] = None
    meta: Dict = {}

class QueryResponse(BaseModel):
    answer: str
    intent: str
    tool: Optional[str] = None
    confidence: float
    sources: List[Source] = []