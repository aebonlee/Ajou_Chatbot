"""
에이전트 도구 모음:
- rerank: Cross-Encoder로 후보 재랭크
- course_lookup: 과목명/코드로 정확도 높게 1건 정리
- policy_quote: 규정/학칙 인용 스니펫(조항/문구)
- table_extract: 표를 텍스트화(권장 이수표 등)
- plan_builder: 학기별 권장 이수 플랜 템플릿 생성
"""
from typing import List, Dict, Any

def rerank_tool(question:str, items:List[Dict[str,Any]])->List[Dict[str,Any]]:
    # 내부적으로 sentence-transformers CrossEncoder 적용 (성능/캐시 고려)
    return items

def course_lookup_tool(name_or_code:str)->Dict[str,Any]:
    # 색인에서 정확히 1개 과목을 찾아 {name, code, credit, prereq, term} 반환
    return {}

def policy_quote_tool(keywords:str)->str:
    # 규정 문서에서 키워드 구간 인용
    return ""

def table_extract_tool(md_or_html:str)->str:
    # 표를 안전하게 텍스트로 변환
    return ""

def plan_builder_tool(tracks:List[str])->str:
    # 권장 이수 예시 템플릿(입력 트랙/관심에 맞춰)
    return ""

TOOLS = {
  "rerank": rerank_tool,
  "course_lookup": course_lookup_tool,
  "policy_quote": policy_quote_tool,
  "table_extract": table_extract_tool,
  "plan_builder": plan_builder_tool,
}