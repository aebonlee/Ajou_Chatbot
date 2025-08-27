from typing import Dict, Any, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.core import config

Category = Literal[
    "major_list","major_detail","micro_list","micro_detail","course_detail",
    "term_plan","track_rules","general_info","rule_info","practice_capstone",
    "area_compare","other"
]

STYLE_GUIDES = {
  "major_list":   "전공/학과명만 간단히 나열하고 세부 과목/학점은 넣지 마세요.",
  "major_detail": "졸업요건/총 이수학점 위주로 요약하고 과목 나열은 1~2개 예시로 제한하세요.",
  "micro_list":   "마이크로전공명만 나열하고 각 전공은 1문장 특징만 요약하세요.",
  "micro_detail": "마이크로전공별 이수학점, 핵심 필수 1과목, 특징 1줄로 요약하세요.",
  "course_detail":"과목명, 학점, 선수과목만 요약하세요. 불필요한 배경 설명은 생략.",
  "general_info": "학과 소개/위치/연락처 등 핵심 정보를 간결히 설명하세요.",
  "rule_info":    "규정·학칙을 인용하고 반드시 출처를 명시하세요.",
  "term_plan":    "학년/학기별 추천 이수 순서를 간단한 불릿으로만 제시하세요.",
  "practice_capstone":"현장실습/인턴/캡스톤 안내를 간단히 설명하고 최신 공지는 학과 공지로 유도하세요.",
  "area_compare": "각 영역을 1~2문장으로 비교 요약하고 장단점을 균형 있게 적어주세요.",
  "other":        "질문 맥락에 맞춰 자연스럽게 답하되 불필요한 나열은 피하세요.",
}

CATEGORY_CONFIG: Dict[str, Dict[str, Any]] = {
    "major_list":     {"lex_weight": 0.9,  "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 30, "assemble_budget_chars": 20000,  "max_ctx_chunks": 4, "max_tokens": 900},
    "major_detail":   {"lex_weight": 0.8,  "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 40, "assemble_budget_chars": 7000,   "max_ctx_chunks": 6, "max_tokens": 1200},
    "micro_list":     {"lex_weight": 0.9,  "micro_mode": "only",    "rerank": True,  "rerank_candidates": 30, "assemble_budget_chars": 12000,  "max_ctx_chunks": 4, "max_tokens": 800},
    "micro_detail":   {"lex_weight": 0.8,  "micro_mode": "only",    "rerank": True,  "rerank_candidates": 40, "assemble_budget_chars": 15000,  "max_ctx_chunks": 5, "max_tokens": 1000},
    "course_detail":  {"lex_weight": 0.7,  "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 50, "assemble_budget_chars": 5000,   "max_ctx_chunks": 5, "max_tokens": 900},
    "term_plan":      {"lex_weight": 0.8,  "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 50, "assemble_budget_chars": 6000,   "max_ctx_chunks": 6, "max_tokens": 1100},
    "track_rules":    {"lex_weight": 0.75, "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 40, "assemble_budget_chars": 12000,  "max_ctx_chunks": 5, "max_tokens": 900},
    "general_info":   {"lex_weight": 0.9,  "micro_mode": "exclude", "rerank": False,                         "assemble_budget_chars": 12000,  "max_ctx_chunks": 4, "max_tokens": 900},
    "rule_info":      {"lex_weight": 0.7,  "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 50, "assemble_budget_chars": 15000,  "max_ctx_chunks": 6, "max_tokens": 1000},
    "practice_capstone":{"lex_weight":0.8, "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 40, "assemble_budget_chars": 12000,  "max_ctx_chunks": 5, "max_tokens": 900},
    "area_compare":   {"lex_weight": 0.85, "micro_mode": "include", "rerank": True,  "rerank_candidates": 40, "assemble_budget_chars": 15000,  "max_ctx_chunks": 6, "max_tokens": 1000},
    "other":          {"lex_weight": 0.8,  "micro_mode": "exclude", "rerank": True,  "rerank_candidates": 30, "assemble_budget_chars": 15000,  "max_ctx_chunks": 6, "max_tokens": 1000},
}

def _apply_category_overrides(state: Dict[str, Any], category: str) -> None:
    cfg = CATEGORY_CONFIG.get(category, {})
    if not cfg: return
    opts = state.setdefault("opts", {})
    for k, v in cfg.items():
        if k == "micro_mode":
            opts[k] = v
        else:
            if opts.get(k) is None:
                opts[k] = v

def _heuristic(q: str) -> str:
    s = q.replace(" ", "").lower()
    if any(k in s for k in ["학과목록","전공목록","무슨학과","학과있","전공있"]): return "major_list"
    if any(k in s for k in ["졸업요건","총이수","권장이수","교육과정","로드맵"]): return "major_detail"
    if "마이크로전공" in s and any(k in s for k in ["뭐","무엇","종류","목록","리스트"]): return "micro_list"
    if "마이크로전공" in s: return "micro_detail"
    if any(k in s for k in ["과목","선수","수업","영어강의","코드","학점","개설학기"]): return "course_detail"
    if any(k in s for k in ["학기별","권장순서","1학기","2학기","학년","무슨과목","뭐들어야","수강"]): return "term_plan"
    if any(k in s for k in ["복수전공","부전공","연계전공","융합전공","전과"]): return "track_rules"
    if any(k in s for k in ["학과소개","연락처","위치","교수","사무실"]): return "general_info"
    if any(k in s for k in ["학칙","규정","조항","제","정원","평점","재수강"]): return "rule_info"
    if any(k in s for k in ["캡스톤","현장실습","인턴","졸업작품"]): return "practice_capstone"
    if any(k in s for k in ["영역","비교","추천","트랙","로드맵"]): return "area_compare"
    return "other"

class RouteSchema(BaseModel):
    primary: Category
    secondary: List[Category] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

ROUTER_SYSTEM = (
    "너는 대학 학사안내 질문을 카테고리로 분류하는 라우터야.\n"
    "가능한 카테고리: major_list, major_detail, micro_list, micro_detail, course_detail,\n"
    "term_plan, track_rules, general_info, rule_info, practice_capstone, area_compare, other\n\n"
    "규칙:\n"
    "- 질문에 가장 적합한 1개를 primary로 고르고, 추가로 해당될 수 있는 것들을 secondary에 넣을 것.\n"
    "- 신입생 자연어/오타/우회표현을 고려할 것.\n"
    "- '무슨 과목', '뭐 들어야', '수강' 등은 term_plan으로 분류.\n"
    "- 신청/절차/포탈 관련이면 'track_rules'. 캡스톤/인턴/현장실습/졸업작품은 'practice_capstone'.\n"
    "- 학칙/규정/조항/정원/재수강 등 제도 인용은 'rule_info'.\n"
    "- 확신도가 낮으면 confidence를 낮게."
)

def _fixed_answer(category: str) -> str:
    if category == "practice_capstone":
        return ("캡스톤·인턴·현장실습·졸업작품 관련 모집은 보통 **학기 시작 전 사전 신청**으로 진행됩니다.\n"
                "최신 일정은 각 학과 공지사항을 확인하거나 학과 사무실로 문의해 주세요.")
    if category == "track_rules":
        return ("복수전공/부전공/연계·융합전공/전과 신청은 **아주대학교 포탈(학사서비스)**에서 진행됩니다.\n"
                "세부 요건과 선발 기준은 소속 단과대·학과 공지를 함께 확인해 주세요.")
    if category == "rule_info":
        return ("학칙/규정에 관한 문의입니다. 공통 학사 규정 페이지를 함께 확인하시면 정확합니다.")
    return ""

def node_classify(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("error"): return state
    q = state["question"]; hcat = _heuristic(q); chosen = hcat
    confidence = 1.0 if hcat != "other" else 0.0
    use_llm = bool(state["opts"].get("use_llm", True))
    need_llm = (hcat == "other") or bool(state["opts"].get("force_llm_route", False))
    if use_llm and need_llm:
        try:
            llm = ChatOpenAI(model=state["opts"].get("model_name", config.LLM_MODEL), temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", ROUTER_SYSTEM),
                ("user", "질문: {q}\nJSON만 출력: {\"primary\": \"...\", \"secondary\": [\"...\"], \"confidence\": 0.0~1.0}"),
            ])
            out = llm.with_structured_output(RouteSchema).invoke(prompt.format_messages(q=q))
            if out and out.primary:
                chosen = out.primary; confidence = float(out.confidence or 0.0)
                if confidence < 0.55 and hcat != "other": chosen = hcat
        except Exception:
            pass
    state["category"] = chosen
    state["style_guide"] = STYLE_GUIDES.get(chosen, STYLE_GUIDES["other"])
    _apply_category_overrides(state, chosen)
    if chosen in ("practice_capstone", "track_rules", "rule_info"):
        state["answer"] = _fixed_answer(chosen); state["skip_rag"] = True
    else:
        state["skip_rag"] = False
    return state