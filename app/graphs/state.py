# app/graphs/state.py
"""
LangGraph 상태 정의 (TypedDict).
- 노드 간 전달되는 모든 키를 한눈에 정리.
"""
from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict, total=False):
    # 입력
    question: str
    user_id: str

    # 옵션/설정
    opts: Dict[str, Any]

    # 파싱 결과 (LLM/휴리스틱에서 추출된 스코프 등)
    context_struct: Dict[str, Any]  # QuerySchema.dict() 형태 저장
    needs_clarification: bool
    clarification_prompt: Optional[str]

    # 분류/라우팅 결과
    category: Optional[str]          # 예: major_detail, micro_list, practice_capstone, ...
    style_guide: Optional[str]       # 카테고리별 스타일 지침 문자열
    skip_rag: bool                   # True면 RAG 스킵(고정응답/외부 포워딩 등)

    # 검색/컨텍스트/답변
    retrieved: List[Dict[str, Any]]  # retriever 출력 딕셔너리 리스트
    context: str
    answer: Optional[str]
    llm_answer: Optional[str]

    # 에러
    error: Optional[str]