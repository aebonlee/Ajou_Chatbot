from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question: str
    departments: List[str] = Field(default_factory=list)
    topk: int = 8
    debug: bool = False
    use_llm: bool = True
    # 선택 옵션(없으면 내부 기본값)
    micro_mode: Literal["exclude","include","only"] = "exclude" # 기본값을 'exclude'로 고정, 관련 키워드 포함시 include로 판별
    assemble_budget_chars: Optional[int] = None
    max_ctx_chunks: Optional[int] = None
    rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_candidates: Optional[int] = None

class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    llm_answer: Optional[str] = None
    context: Optional[str]
    sources: List[str] = Field(default_factory=list)
    micro_mode: str
    error: Optional[str] = None
    clarification: Optional[str] = None

# 내부 LLM 구조화용
class QuerySchema(BaseModel):
    faculties: List[str] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None)
    need_slots: List[str] = Field(default_factory=list)

# (선택) 라우터 스키마도 노출해 두면 재사용 가능
Category = Literal[
  "major_list","major_detail","micro_list","micro_detail","course_detail",
  "term_plan","track_rules","general_info","rule_info","practice_capstone",
  "area_compare","other"
]

class RouteSchema(BaseModel):
    primary: Category
    secondary: List[Category] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


# --------------------------------------------
# 학사공통
# -------------------------------------------
INFO_NAME_MAP = {
    "학칙": "rules",
    "학사력": "overview",
    "대학생활안내": "campus_life"
}


class InfoRequest(BaseModel):
    question: str
    departments: List[str] = []  # corpus_type (rules, overview, campus_life)
    selected_list: List[str] = []

    @field_validator('departments', mode='before')
    @classmethod
    def translate_departments(cls, value: List[str]) -> List[str]:
        translated_list = []
        for kor_name in value:
            system_id = INFO_NAME_MAP.get(kor_name)
            if system_id is None:
                raise ValueError(f"'{kor_name}'은(는) 유효한 department가 아닙니다.")
            translated_list.append(system_id)
        return translated_list

class InfoResponse(BaseModel):
    answer: str
    question: str


class NoticeQuery(BaseModel):
    question: str

class NoticeResponse(BaseModel):
    answer: str
