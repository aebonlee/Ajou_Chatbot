# app/graphs/state.py
from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict, total=False):
    question: str
    user_id: str
    opts: Dict[str, Any]

    context_struct: Dict[str, Any]
    needs_clarification: bool
    clarification_prompt: Optional[str]

    category: Optional[str]
    style_guide: Optional[str]
    skip_rag: bool

    retrieved: List[Dict[str, Any]]
    context: str
    answer: Optional[str]
    llm_answer: Optional[str]

    error: Optional[str]
    must_include: List[str]