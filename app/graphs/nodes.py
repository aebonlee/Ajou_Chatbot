"""
LangGraph에서 사용하는 '노드' 모음.

그래프 단계 개요
1) node_parse_intent   : 질문에서 단과대/학과/학년 등 구조화 정보 추출(LLM or 휴리스틱)
2) node_need_more      : 학과/단과대 추출 실패 시 추가정보 요구 플래그 설정
3) node_retrieve       : 벡터/용어 혼합 검색 → 후보 문서(hit) 목록 반환
4) node_build_context  : hit 목록을 컨텍스트 텍스트로 구성(예산/조각 수 제한)
5) node_classify       : 질문 카테고리 분류(간단 휴리스틱 + 옵션으로 LLM)
6) node_answer         : 최종 답변 생성(인사말/본문/출처 병합 및 후처리)

역할 분리 원칙
- '인사말 생성', '출처 1회만 출력', '질문 에코 제거' 같은 UX 후처리는 여기서 책임진다.
- 서버는 결과를 있는 그대로 전달만 한다.
"""

from typing import List, Dict, Any, Literal
import json, re, random, os
from app.services.retriever import (
    get_cached_retrievers,
    get_all_cached_chunks,
    get_filtered_bm25_retriever,
    weighted_reciprocal_rank_fusion,
    format_docs,
)
from app.core.config import rag_logger

# LLM 클라이언트 (사용 가능한 것만 import)
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core import config
from app.models.schemas import QuerySchema         # LLM 구조화 출력 스키마
from app.services.retriever import retrieve        # 검색기(벡터+용어)
from app.services.textutil import term_sort_key    # 학기 정렬용 키 생성

# -----------------------------------------------------------------------------
# 인사말/토픽 유틸
# -----------------------------------------------------------------------------
INTRO_TEMPLATES_WITH_TOPIC = [
    "{topic} 관련 핵심만 정리해 드릴게요.",
    "{topic} 질문 주셔서 감사합니다. 요점만 안내드릴게요.",
    "{topic}에 대해 물어보셨네요. 바로 정리해 드립니다.",
    "{topic}: 아래에 한눈에 보이게 정리했어요.",
]
INTRO_TEMPLATES_GENERIC = [
    "좋은 질문이에요! 아래에 핵심만 정리해 드릴게요.",
    "문의 감사합니다. 요점만 빠르게 정리해 드립니다.",
]

# 학과별 연락처 및 소속 정보 데이터
DEPARTMENT_INFO = {
    # 공과대학 소속
    "건설시스템공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2520"]
    },
    "건축학과": {
        "college": "공과대학",
        "contacts": ["031-219-2410"]
    },
    "교통시스템공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2530"]
    },
    "기계공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2340"]
    },
    "미래자동차연계전공": {
        "college": "공과대학",
        "contacts": ["031-219-2340"]  # 기계공학과와 동일하다고 가정
    },
    "산업공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2350"]
    },
    "융합시스템공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2360"]
    },
    "응용화학과": {
        "college": "공과대학",
        "contacts": ["031-219-2370"]
    },
    "첨단신소재공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2380"]
    },
    "화학공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2390"]
    },
    "환경안전공학과": {
        "college": "공과대학",
        "contacts": ["031-219-2400"]
    },

    # 소프트웨어융합대학 소속
    "국방디지털융합학과": {
        "college": "소프트웨어융합대학",
        "contacts": ["031-219-2640"]
    },
    "데이터보안·활용융합 연계전공": {
        "college": "소프트웨어융합대학",
        "contacts": ["031-219-2650"]
    },
    "디지털미디어학과": {
        "college": "소프트웨어융합대학",
        "contacts": ["031-219-2630"]
    },
    "사이버보안학과": {
        "college": "소프트웨어융합대학",
        "contacts": ["031-219-2610"]
    },
    "소프트웨어학과": {
        "college": "소프트웨어융합대학",
        "contacts": [
            "수업(수강신청), 학적관리(전과 등): 031-219-2431",
            "졸업요건, 현장실습과목: 031-219-1687",
            "장학, 자기주도과목, 학생행사: 031-219-2430"
        ]
    },
    "인공지능융합학과": {
        "college": "소프트웨어융합대학",
        "contacts": ["031-219-2620"]
    },
    "인문사회데이터분석전공": {
        "college": "소프트웨어융합대학",
        "contacts": ["031-219-2660"]
    }
}

# 단과대별 요람 링크
COLLEGE_BULLETIN_LINKS = {
    "공과대학": "https://www.ajou.ac.kr/kr/bachelor/bulletin.do?mode=list&srSearchKey=&srCategoryId=79&srSearchVal=",
    "소프트웨어융합대학": "https://www.ajou.ac.kr/kr/bachelor/bulletin.do?mode=list&srSearchKey=&srCategoryId=80&srSearchVal="
}


_KOR_WS = re.compile(r"\s+")
_DEPT_PAT = re.compile(r"([가-힣A-Za-z0-9]+학과)")
_TERM_PAT = re.compile(r"([1-4])\s*학\s*년(?:\s*([1-2])\s*학\s*기)?")
_KEYWORDS = ["전공필수", "졸업요건", "이수학점", "권장 이수", "교육과정", "과목", "선수과목", "BSM", "학기별", "로드맵"]

def _compact_spaces(s: str) -> str:
    """연속 공백/개행을 1칸으로 압축."""
    return _KOR_WS.sub(" ", (s or "").strip())

def _trim_topic(topic: str, max_len: int = 22) -> str:
    """UI에 들어갈 짧은 토픽 문자열로 다듬기."""
    t = topic.strip(" \n\t-–—:·,.\"'“”‘’")
    return t if len(t) <= max_len else (t[:max_len-1] + "…")

def _extract_topic(question: str, state: Dict[str, Any]) -> str:
    """
    질문에서 UI용 '짧은 토픽'을 추출.
    우선순위: (스코프 힌트) → ('OO학과' 패턴) → (학년/학기) → (핵심 키워드).
    """
    q = _compact_spaces(question or "")

    # 스코프 힌트/파싱 결과에서 학과명 우선 사용
    depts = list(state.get("context_struct", {}).get("departments") or []) \
            or list(state.get("opts", {}).get("scope_depts") or [])
    dept_name = (depts[0] if depts else "")

    # 질문 문장 자체에서 'OO학과' 캡처
    if not dept_name:
        m = _DEPT_PAT.search(q)
        if m: dept_name = m.group(1)

    # 학년/학기
    term_txt = ""
    m = _TERM_PAT.search(q)
    if m:
        y, s = m.group(1), m.group(2)
        term_txt = f"{y}학년 {s}학기" if (y and s) else (f"{y}학년" if y else "")

    # 주요 키워드(2개까지)
    hits = [kw for kw in _KEYWORDS if kw in q]
    key_part = " · ".join(hits[:2]) if hits else ""

    parts = [p for p in (dept_name, term_txt, key_part) if p]
    if parts:
        return _trim_topic(" | ".join(parts) if len(parts) >= 2 else parts[0])

    # 폴백: 질문 축약
    fallback = re.sub(r"[\"'“”‘’]", "", q)
    fallback = re.sub(r"(에 대해|만|으로|는|은|이|가|을|를|요)$", "", fallback)
    return _trim_topic(fallback) if fallback else ""

def _pick_intro(question: str, state: Dict[str, Any]) -> str:
    """토픽이 있으면 토픽형 인사, 없으면 일반 인사."""
    t = _extract_topic(question, state)
    return (random.choice(INTRO_TEMPLATES_WITH_TOPIC).format(topic=t)) if t else random.choice(INTRO_TEMPLATES_GENERIC)


def _extract_departments_from_state(state: Dict[str, Any]) -> List[str]:
    """state에서 학과명 목록을 추출"""
    departments = []

    # context_struct에서 추출
    ctx = state.get("context_struct", {})
    if ctx.get("departments"):
        departments.extend(ctx["departments"])

    # opts에서 scope_depts 추출
    opts = state.get("opts", {})
    if opts.get("scope_depts"):
        departments.extend(opts["scope_depts"])

    # 중복 제거 및 정규화
    unique_depts = list(set(departments))
    return [dept for dept in unique_depts if dept in DEPARTMENT_INFO]


def _generate_department_info_appendix(departments: List[str]) -> str:
    """학과 정보 부록 생성"""
    if not departments:
        return ""

    # 단과대별로 그룹핑
    colleges_info = {}
    for dept in departments:
        if dept not in DEPARTMENT_INFO:
            continue

        dept_info = DEPARTMENT_INFO[dept]
        college = dept_info["college"]

        if college not in colleges_info:
            colleges_info[college] = []
        colleges_info[college].append(dept)

    if not colleges_info:
        return ""

    appendix_parts = [
        "덧붙여서 더 자세한 정보는 아래의 요람 문서를 확인하시거나, 학과사무실에 연락하시면 더 정확하고 상세한 정보를 얻으실 수 있어요!"
    ]

    # 요람 링크 추가
    appendix_parts.append("\n【아주대학교 요람】")
    for college, depts in colleges_info.items():
        if college in COLLEGE_BULLETIN_LINKS:
            appendix_parts.append(f"* {college}: {COLLEGE_BULLETIN_LINKS[college]}")

    # 연락처 정보 추가
    appendix_parts.append("\n【학과사무실 연락처】")
    for college, depts in colleges_info.items():
        for dept in sorted(depts):  # 학과명 정렬
            dept_info = DEPARTMENT_INFO[dept]
            contacts = dept_info["contacts"]

            appendix_parts.append(f"* {dept}")
            for contact in contacts:
                if ":" in contact:  # 구체적인 업무 구분이 있는 경우
                    appendix_parts.append(f"   * {contact}")
                else:  # 단순 전화번호
                    appendix_parts.append(f"   * {contact}")

    return "\n".join(appendix_parts)


# -----------------------------------------------------------------------------
# 공용 텍스트/컨텍스트 유틸
# -----------------------------------------------------------------------------
def _safe_path(h: Dict[str, Any]) -> str:
    """히트의 경로나 메타에 기재된 경로를 추출."""
    return (h.get("path") or (h.get("metadata") or {}).get("path") or "").strip()

def _dedup_lines(text: str) -> str:
    """바로 위 줄과 동일한 내용은 제거(스크랩 중복 줄 방지)."""
    out, prev = [], None
    for ln in (text or "").splitlines():
        cur = ln.strip()
        if cur and cur == prev:
            continue
        out.append(ln)
        prev = cur
    return "\n".join(out)

def _summarize_sources(hits: List[Dict[str, Any]], max_items: int = 8) -> str:
    """
    UI에 들어갈 '출처 요약' 문자열을 만든다.
    - 파일 이름/제목 위주로 간결하게.
    """
    if not hits:
        return "(없음)"
    seen_sources = set()
    unique_sources = []

    for h in hits[:max_items * 2]:  # 더 많이 확인해서 중복 제거
        meta = h.get("metadata") or {}
        path = _safe_path(h)

        # 경로에서 핵심 부분만 추출
        if " > " in path:
            parts = path.split(" > ")
            if len(parts) >= 3:
                # "소프트웨어융합대학 > 디지털미디어학과 > 디지털미디어전공"
                # -> "디지털미디어학과 디지털미디어전공"
                simplified = f"{parts[1]} {parts[2]}"
            else:
                simplified = parts[-1]  # 마지막 부분만
        else:
            simplified = os.path.basename(path) or path

        simplified = simplified.replace("_", " ")

        if simplified not in seen_sources and len(unique_sources) < max_items:
            seen_sources.add(simplified)
            unique_sources.append(simplified)

    if len(unique_sources) == 1:
        return f'자세한 내용은 학사요람의 "{unique_sources[0]}" 항목에서 확인할 수 있습니다.'
    elif len(unique_sources) <= 3:
        return f'자세한 내용은 학사요람의 "{", ".join(unique_sources)}" 항목 등에서 확인할 수 있습니다.'
    else:
        return f'자세한 내용은 학사요람의 "{unique_sources[0]}" 항목 등 관련 문서에서 확인할 수 있습니다.'

def _build_context_from_hits(hits: List[Dict[str, Any]], *, max_items: int, budget_chars: int) -> str:
    """
    LLM에 투입할 컨텍스트 텍스트를 조립한다.
    - 연/학기 메타가 있는 경우 우선 정렬
    - 예산(budget_chars) 안에서 출처별 블록을 이어붙임
    """
    if not hits:
        return ""
    def _ord(h):
        m = h.get("metadata") or {}
        y, s = m.get("year"), m.get("semester")
        return (0, *term_sort_key(y, s)) if (y or s) else (1, 99, 99)

    hits = sorted(hits, key=_ord)

    parts, used = [], 0
    for i, h in enumerate(hits[:max_items], 1):
        path = _safe_path(h)
        body = _dedup_lines(h.get("document") or "")
        block = f"[SOURCE {i}] {path}\n{body}\n"
        blen = len(block)
        if used + blen > budget_chars:
            remain = budget_chars - used
            if remain > 200:  # 최소 블록 길이 보장
                parts.append(block[:remain])
            break
        parts.append(block)
        used += blen
    return "\n\n---\n\n".join(parts)

def _make_llm(model_name: str, temperature: float, max_tokens: int):
    """모델명으로 OpenAI/Anthropic 중 적절한 클라이언트를 생성."""
    provider = config.llm_provider_from_model(model_name)
    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic 미설치 또는 로드 실패")
        return ChatAnthropic(model_name=model_name, temperature=temperature, timeout=None, stop=None)
    else:
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai 미설치 또는 로드 실패")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)


# -----------------------------------------------------------------------------
# 노드: 의도 파싱 → 추가정보 필요 여부 → 검색 → 컨텍스트 조립
# -----------------------------------------------------------------------------
def node_parse_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문에서 단과대/학과/학년 등 구조화 정보를 뽑는다.
    - use_llm=False면 힌트(scope_depts)만 그대로 사용
    - use_llm=True면 LLM으로 JSON 구조화
    """
    use_llm = bool(state["opts"].get("use_llm", True))
    hints_depts = state["opts"].get("scope_depts") or []
    if "micro_mode" not in state["opts"]:
        state["opts"]["micro_mode"] = None

    if not use_llm:
        state["context_struct"] = {"faculties": [], "departments": hints_depts, "year": None, "need_slots": []}
        return state

    q = state["question"]
    try:
        model_name = state["opts"].get("model_name", config.LLM_MODEL)
        llm = _make_llm(
            model_name=model_name,
            temperature=float(state["opts"].get("temperature") or 0.0),
            max_tokens=int(state["opts"].get("max_tokens") or 400),
        )
        provider = config.llm_provider_from_model(model_name)

        if provider == "anthropic":
            # Claude 계열: JSON 문자열을 직접 파싱
            sys = ("너는 대학 학사요람 Q&A용 추출기야. "
                   '입력에서 {"faculties":[],"departments":[],"year":null,"need_slots":[]} JSON만 출력.')
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                ("user", "질문: {q}\nJSON만 출력"),
            ])
            out = llm.invoke(prompt.format_messages(q=q))
            raw = (out.content or "").strip()
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.S)
                if m:
                    parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                qs = QuerySchema(
                    faculties=parsed.get("faculties") or [],
                    departments=parsed.get("departments") or [],
                    year=parsed.get("year"),
                    need_slots=parsed.get("need_slots") or [],
                )
            else:
                raise ValueError("parse_intent: JSON 파싱 실패")
        else:
            # OpenAI 계열: LangChain의 구조화 출력 사용
            sys = "너는 대학 학사요람 Q&A용 추출기다. 단과대/학과/학년도/필요 슬롯을 구조화하여 JSON으로 돌려라."
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                ("user", "질문: {q}\nkeys: faculties, departments, year, need_slots"),
            ])
            out = llm.with_structured_output(QuerySchema).invoke(prompt.format_messages(q=q))
            qs = out

        depts_union = list({*(qs.departments or []), *hints_depts})
        state["context_struct"] = {
            "faculties": list(qs.faculties or []),
            "departments": depts_union,
            "year": qs.year,
            "need_slots": list(qs.need_slots or []),
        }
        return state

    except Exception:
        # LLM 실패 시 힌트만 반영
        state["context_struct"] = {"faculties": [], "departments": hints_depts, "year": None, "need_slots": []}
        return state

def node_need_more(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    학과/단과대 정보가 하나도 없으면 '추가 설명 필요' 플래그를 세움.
    프런트는 clarification 프롬프트를 표시해 추가 정보를 유도할 수 있다.
    """
    if not bool(state["opts"].get("use_llm", True)):
        state["needs_clarification"] = False
        return state
    ctx = state["context_struct"]
    state["needs_clarification"] = not (ctx.get("departments") or ctx.get("faculties"))
    if state["needs_clarification"]:
        state["clarification_prompt"] = "어느 학과(또는 전공) 기준인지 알려주세요. 예) 디지털미디어학과 / 소프트웨어학과"
    return state

def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    검색 단계: lex(용어) + vector 혼합 검색으로 문서 히트 목록을 가져온다.
    - rerank 옵션 사용 시 Cross-Encoder로 재정렬
    """
    if state.get("needs_clarification") or state.get("error") or state.get("skip_rag"):
        state["retrieved"] = []
        return state

    ctx, opts = state["context_struct"], state["opts"]
    try:
        hits = retrieve(
            state["question"],
            persist_dir=opts["persist_dir"],
            collection=opts["collection"],
            embedding_model=opts["embedding_model"],
            topk=int(opts.get("topk") or config.TOPK),
            lex_weight=float(opts.get("lex_weight") or config.LEX_WEIGHT),
            scope_colleges=(ctx.get("faculties") or None),
            scope_depts=(ctx.get("departments") or None),
            micro_mode=opts.get("micro_mode"),
            debug=bool(opts.get("debug") or False),
            rerank=bool(opts.get("rerank") or False),
            rerank_model=opts.get("rerank_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_candidates=int(opts.get("rerank_candidates") or 30),
            stitch_by_path=False,  # 경로 단위 확장 대신 섹션 조각 사용
        )
        state["retrieved"] = hits
        return state
    except Exception as e:
        state["error"] = f"retrieval_error: {e}"
        state["retrieved"] = []
        return state

def node_build_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    히트 목록을 LLM 입력으로 사용할 컨텍스트 문자열로 조립한다.
    - assemble_budget_chars / max_ctx_chunks로 길이 제어
    """
    if state.get("error") or state.get("skip_rag"):
        state["context"] = ""
        return state

    hits, opts = state.get("retrieved") or [], state["opts"]
    state["context"] = _build_context_from_hits(
        hits,
        max_items=int(opts.get("max_ctx_chunks") or 8),
        budget_chars=int(opts.get("assemble_budget_chars") or 60000),
    )
    state["must_include"] = []  # 필요 시 '반드시 포함할 스니펫' 지정
    return state


# -----------------------------------------------------------------------------
# 라우팅/스타일 가이드 + 후처리(인사/출처/문장 정리)
# -----------------------------------------------------------------------------
Category = Literal[
    "major_list","major_detail","micro_list","micro_detail","course_detail",
    "term_plan","track_rules","general_info","rule_info","practice_capstone",
    "area_compare","other"
]

STYLE_GUIDES = {
  # 필요한 카테고리만 우선 문서화(확장 용이)
  "major_detail": "졸업요건/총 이수학점 위주 요약. 과목 나열은 최소화.",
  "term_plan":    "학년/학기별 권장 순서를 불릿으로 간단히.",
  "rule_info":    "학칙·규정을 근거로 인용. 반드시 출처 포함.",
  "other":        "간결/불필요한 서론 금지.",
}

def _heuristic(q: str) -> str:
    """휴리스틱 분류(간단 규칙). 확장 시 LLM 라우터와 혼합 가능."""
    s = q.replace(" ", "").lower()
    if any(k in s for k in ["졸업요건","총이수","교육과정","로드맵"]): return "major_detail"
    if any(k in s for k in ["학기별","권장순서","학년","뭐들어야","수강"]): return "term_plan"
    if any(k in s for k in ["학칙","규정","조항","정원","재수강"]): return "rule_info"
    return "other"

def node_classify(state: Dict[str, Any]) -> Dict[str, Any]:
    """질문 카테고리를 정하고, 카테고리별 기본 옵션/스타일을 주입."""
    hcat = _heuristic(state["question"])
    state["category"] = hcat
    state["style_guide"] = STYLE_GUIDES.get(hcat, STYLE_GUIDES["other"])
    state["skip_rag"] = False  # 지금은 모든 카테고리에 대해 RAG 수행
    return state


# --- 후처리 유틸(중복 인사/따옴표 에코/출처 섹션 병합) -----------------------
_QUOTED_LINE_RE = re.compile(r'^\s*[\"“”].*[\"“”]\s*$')
_SRC_SPLIT_RE = re.compile(r"\n\s*출처\s*:\s*\n", re.I)

def _strip_redundant_lead(text: str) -> str:
    """첫 줄에 따옴표로 질문을 다시 쓰는 패턴 등 제거."""
    if not text:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    if _QUOTED_LINE_RE.match(first) or first.endswith("에 대해 질문해 주셨군요!"):
        return "\n".join(lines[1:]).strip()
    return text

def _merge_sources(text: str) -> str:
    """
    모델이 실수로 '출처:' 섹션을 여러 번 생성하는 경우 1개로 병합.
    """
    if not text:
        return text
    parts = _SRC_SPLIT_RE.split(text)
    if len(parts) <= 2:
        return text
    body = parts[0].rstrip()
    tails = [p.strip() for p in parts[1:] if p.strip()]
    merged = []
    for t in tails:
        for ln in t.splitlines():
            ln = ln.strip()
            if ln and ln not in merged:
                merged.append(ln)
    return f"{body}\n\n출처:\n" + "\n".join(merged)


def _check_response_completeness(text: str) -> bool:
    """마침표/종결어미, 출처 존재 여부로 '완성도' 대략 점검."""
    text = (text or "").strip()
    if not text:
        return False

    # 학과 정보 부록이 있으면 완성된 것으로 간주
    if '【학과사무실 연락처】' in text or '【아주대학교 요람】' in text:
        return True

    if '출처:' in text or '출처\n' in text:
        return True

    return text.endswith(('.', '요.', '다.', '니다.', '습니다.', '세요.', '어요.', '아요.', '지요.'))


# -----------------------------------------------------------------------------
# 답변 생성 노드
# -----------------------------------------------------------------------------
def node_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    최종 답변 생성 단계.
    - LLM 비활성/히트 없음: 간단 안내 + 출처 요약
    - LLM 활성: 시스템/사용자 프롬프트 구성 → 모델 호출 → 후처리
    - 특정 카테고리의 경우 학과 정보 부록 추가
    """
    if state.get("skip_rag") or state.get("needs_clarification") or state.get("error"):
        return state

    hits = state.get("retrieved") or []
    use_llm = bool(state["opts"].get("use_llm", True))
    category = state.get("category", "")

    # 학과 정보 부록을 추가할 카테고리들
    dept_info_categories = {"major_list", "major_detail", "micro_list", "micro_detail", "course_detail", "term_plan"}
    should_add_dept_info = category in dept_info_categories

    # --- LLM 미사용 또는 히트 없음 → 기본 안내 + 출처 요약
    if not use_llm or not hits:
        src = _summarize_sources(hits)
        body = "검색된 문서 요약을 제공합니다."
        final_txt = body
        if "출처:" not in final_txt:
            final_txt += "\n\n출처:\n" + src

        # 학과 정보 부록 추가
        if should_add_dept_info:
            departments = _extract_departments_from_state(state)
            dept_appendix = _generate_department_info_appendix(departments)
            if dept_appendix:
                final_txt += "\n\n" + dept_appendix

        state["llm_answer"] = final_txt
        state["answer"] = final_txt
        return state

    # --- LLM 사용 경로 (기존 코드 동일)
    micro_mode = state["opts"].get("micro_mode", "exclude")
    style_guide = state.get("style_guide") or ""
    rule = {
        "exclude": "1) 마이크로전공 내용은 제외하고 본전공 중심으로 답하세요.",
        "only": "1) 마이크로전공만 대상으로 답하세요.",
        "include": "1) 본전공과 마이크로를 모두 포함하되 본전공을 우선하세요.",
    }.get(micro_mode, "1) 본전공을 우선하되 필요 시 마이크로도 포함하세요.")

    persona = (
        "아주대학교의 친근하고 도움이 되는 학사안내 도우미입니다. "
        "존댓말로 친근하게 답변하며, '네, ~에 대해 간략히 설명해 드릴게요!' 같은 인사말로 시작합니다."
    )

    sys = (
        f"{persona}\n"
        f"답변 형식:\n"
        f"- '네, [질문 내용]에 대해 간략히 설명해 드릴게요!' 로 시작\n"
        f"- CONTEXT 근거로만 답변하고, 근거가 없으면 '문서에서 확인되지 않습니다'라고 명시\n"
        f"- {rule}\n"
        f"- 가이드: {style_guide}\n"
        f"- 불릿 포인트와 숫자 목록을 활용해 보기 쉽게 구성\n"
        f"- 답변 내용만 작성하고 별도 출처 섹션은 추가하지 마세요"
    )

    # 컨텍스트 길이 방어
    context = state['context']
    if len(context) > 50_000:
        context = context[:45_000] + "\n...(내용이 길어 일부 생략)..."

    usr = (
        f"질문: {state['question']}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"요구사항:\n"
        f"- '네, [질문내용]에 대해 간략히 설명해 드릴게요!' 로 친근하게 시작\n"
        f"- 불릿 포인트나 숫자 목록으로 구조화\n"
        f"- 내용만 작성하고 출처는 따로 추가하지 마세요"
    )

    llm = _make_llm(
        model_name=state["opts"].get("model_name", config.LLM_MODEL),
        temperature=float(state["opts"].get("temperature") or config.TEMPERATURE),
        max_tokens=int(state["opts"].get("max_tokens") or 1500),
    )

    try:
        out = llm.invoke([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}])
        body = (out.content or "").strip()

        # 후처리: 질문 에코 제거 + 중복 '출처:' 병합
        body = _merge_sources(_strip_redundant_lead(body))

        final_txt = body

        # 안전장치: 출처 누락 시 요약 출처 부착 (단, "자세한 내용은"이 이미 있으면 생략)
        if "출처:" not in final_txt and "자세한 내용은" not in final_txt:
            final_txt += "\n\n" + _summarize_sources(hits)

        # 학과 정보 부록 추가
        if should_add_dept_info:
            departments = _extract_departments_from_state(state)
            dept_appendix = _generate_department_info_appendix(departments)
            if dept_appendix:
                final_txt += "\n\n" + dept_appendix

        # 완성도 보정을 부록 추가 후에 실행
        if not _check_response_completeness(final_txt):
            if not final_txt.endswith(("...", "…")):
                final_txt += "..."
            final_txt += "\n\n[응답이 일부 생략되었을 수 있습니다]"

        state["llm_answer"] = final_txt
        state["answer"] = final_txt
        return state

    except Exception as e:
        # 모델 호출 실패 시에도 출처 요약 포함
        state["error"] = f"llm_error: {e}"
        fallback = "모델 호출에 실패했습니다.\n\n" + _summarize_sources(hits)

        # 학과 정보 부록 추가 (오류 상황에서도)
        if should_add_dept_info:
            departments = _extract_departments_from_state(state)
            dept_appendix = _generate_department_info_appendix(departments)
            if dept_appendix:
                fallback += "\n\n" + dept_appendix

        state["answer"] = fallback
        state["llm_answer"] = None
        return state

#--------------------------------------------
# 학사공통
# -------------------------------------------


RETRIEVER_TOP_K = 7
CONFIDENCE_THRESHOLD = 0.5

def retrieve_node(state: Dict[str, Any]):
    question = state["question"]
    departments = state.get("departments")
    user_selected_list = state.get("user_selected_list")
    rag_logger.info(f"--- Retrieving for: '{question}' in {departments} ---")

    enhanced_question = question
    if user_selected_list:
        context_keywords = ", ".join(user_selected_list)
        enhanced_question = f"{context_keywords}에 대한 질문: {question}"

    chroma_r = get_cached_retrievers()
    search_kwargs = {"k": 10}
    if departments:
        if len(departments) == 1:
            filter_query = {"source": departments[0]}
        else:
            filter_query = {"$or": [{"source": dept} for dept in departments]}
        search_kwargs["filter"] = filter_query

    chroma_results = chroma_r.invoke(enhanced_question, **search_kwargs)
    all_chunks = get_all_cached_chunks()
    bm25_r = get_filtered_bm25_retriever(all_chunks, departments)
    bm25_results = bm25_r.invoke(enhanced_question)

    fused_results = weighted_reciprocal_rank_fusion(
        [chroma_results, bm25_results], weights=[0.5, 0.5], c=0
    )

    if not fused_results:
        return {"documents": [], "retrieval_success": False, "top_score": 0.0}

    return {"documents": fused_results, "retrieval_success": True, "top_score": fused_results[0][1]}


def generate_node(state: Dict[str, Any]):
    rag_logger.info("--- Generating Answer ---")
    context = format_docs([doc for doc, _ in state["documents"][:RETRIEVER_TOP_K]])
    user_selected_list = state.get("user_selected_list", [])

    # labels가 있을 경우, LLM에게 추가적인 힌트를 줌
    user_focus_prompt = ""
    if user_selected_list:
        user_focus_prompt = (f"참고: 사용자는 특히 다음 주제에 관심이 있습니다: "
                             f"{', '.join(user_selected_list)}. 이 주제와 관련된 내용을 중심으로 답변을 구성하세요.")
    system_prompt = (
        "당신은 아주대학교의 친근하고 도움이 되는 학사안내 도우미입니다. "
        "한국어로 친근하게 답변하며, 제공된 문서를 기반으로 정확한 정보를 안내합니다."
        "주어진 컨텍스트에 학부생과 대학원의 규정이 모두 있다면, '학부생'의 규정을 중심으로 답변해야 합니다."
    )

    user_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", """
다음 문서들을 바탕으로 질문에 답변해주세요:

<context>
{context}
</context>

사용자 질문: {question}
{user_focus}

답변 형식:
- "네, [질문 내용]에 대해 간략히 설명해 드릴게요!" 로 친근하게 시작
- 답변은 반드시 <context>에 있는 정보만을 사용
- 번호나 불릿 포인트로 구조화된 답변 제공
- 구체적인 수치나 조건이 있으면 명확히 제시
- 답변의 맨 마지막에 답변의 근거가 된 출처를 종합하여 "더 자세한 내용은 [문서명] [페이지]에서 확인하실 수 있어요!" 형식의 한 문장으로 안내해야 합니다.
- 컨텍스트에서 답변을 찾을 수 없다면, "죄송합니다, 제공된 문서에서는 질문에 대한 정보를 찾을 수 없습니다."라고만 답변하세요.
        """)
    ])

    llm = _make_llm(
        model_name=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    chain = user_prompt | llm
    result = chain.invoke({"context": context, "question": state["question"], "user_focus": user_focus_prompt})
    return {"answer": result.content}

def fallback_node(state: Dict[str, Any]):
    rag_logger.info("--- Fallback Triggered ---")
    reason = state.get("fallback_reason", "예상치 못한 오류입니다.")
    return {"answer": reason}

def should_generate(state: Dict[str, Any]) -> str:
    if not state.get("retrieval_success"):
        rag_logger.info("--- Decision: No documents found, routing to fallback. ---")
        state["fallback_reason"] = "관련 문서를 찾을 수 없습니다."
        return "fallback"
    top_score = state.get("top_score", 0.0)
    rag_logger.info(f"--- Top Score: {top_score:.4f} ---")
    if top_score < CONFIDENCE_THRESHOLD:
        rag_logger.info(f"--- Decision: Score below threshold, routing to fallback. ---")
        state["fallback_reason"] = f"관련성이 높은 문서를 찾지 못해 답변하기 어렵습니다. (신뢰도: {top_score:.3f})"
        return "fallback"
    else:
        rag_logger.info("--- Decision: Score is sufficient, routing to generate. ---")
        return "generate"