# app/graphs/state.py
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

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


# --------------------------------------------
# 학사공통
# -------------------------------------------
class GraphStateInfo(TypedDict, total=False):
    # 필수 입력
    question: str
    departments: List[str]
    user_selected_list: List[str]

    # 검색 결과 및 신뢰도
    documents: List[Tuple[Document, float]]
    retrieval_success: bool
    top_score: float
    fallback_reason: str

    answer: str