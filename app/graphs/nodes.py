# app/graphs/nodes.py
from typing import List, Dict, Any
import json, re, random

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None
from langchain_core.prompts import ChatPromptTemplate

from app.core import config
from app.models.schemas import QuerySchema
from app.services.retriever import retrieve
from app.services.textutil import term_sort_key


# -------------------- 인사말 템플릿 & 토픽 추출 --------------------

# {topic} 자리표시자를 사용하는 짧고 자연스러운 템플릿들
INTRO_TEMPLATES_WITH_TOPIC = [
    "{topic} 관련 핵심만 정리해 드릴게요.",
    "{topic} 질문 주셔서 감사합니다. 요점만 안내드릴게요.",
    "{topic}에 대해 물어보셨네요. 바로 정리해 드립니다.",
    "좋은 질문이에요—{topic} 기준으로 안내드릴게요.",
    "{topic}: 아래에 한눈에 보이게 정리했어요.",
    "{topic} 중심으로 필요한 것만 깔끔히 정리했습니다.",
]

# 토픽을 뽑지 못했을 때 사용할 기본 템플릿
INTRO_TEMPLATES_GENERIC = [
    "좋은 질문이에요! 아래에 핵심만 정리해 드릴게요.",
    "문의 감사합니다. 요점만 빠르게 정리해 드립니다.",
    "바로 핵심만 안내드릴게요.",
]

_KOR_WS = re.compile(r"\s+")
_DEPT_PAT = re.compile(r"([가-힣A-Za-z0-9]+학과)")
_TERM_PAT = re.compile(r"([1-4])\s*학\s*년(?:\s*([1-2])\s*학\s*기)?")
_KEYWORDS = [
    "전공필수", "졸업요건", "졸업 이수학점", "이수학점", "권장 이수", "교육과정",
    "과목", "선수과목", "BSM", "학기별", "로드맵", "전공기초", "영어강의",
]

def _compact_spaces(s: str) -> str:
    return _KOR_WS.sub(" ", (s or "").strip())

def _trim_topic(topic: str, max_len: int = 22) -> str:
    t = topic.strip(" \n\t-–—:·,.\"'“”‘’")
    if len(t) <= max_len:
        return t
    return t[:max_len-1] + "…"

def _extract_topic(question: str, state: Dict[str, Any]) -> str:
    """
    질문에서 짧은 키워드 토픽을 뽑아낸다.
    우선순위:
      1) scope_depts/LLM 파싱 결과의 departments
      2) 질문에서 'OO학과' 패턴
      3) 학년/학기 & 주요 키워드 결합
      4) 주요 키워드 단독
    """
    q = _compact_spaces(question or "")

    # 1) 요청 힌트/파싱 결과로 들어온 학과
    depts = []
    try:
        depts = list(state.get("context_struct", {}).get("departments") or []) \
                or list(state.get("opts", {}).get("scope_depts") or [])
    except Exception:
        depts = []
    dept_name = (depts[0] if depts else "")

    # 2) 질문에서 'OO학과' 직접 추출
    if not dept_name:
        m_dept = _DEPT_PAT.search(q)
        if m_dept:
            dept_name = m_dept.group(1)

    # 3) 학년/학기 추출
    term_txt = ""
    m_term = _TERM_PAT.search(q)
    if m_term:
        y = m_term.group(1)
        s = m_term.group(2)
        if y and s:
            term_txt = f"{y}학년 {s}학기"
        elif y:
            term_txt = f"{y}학년"

    # 4) 주요 키워드 스캔
    hits = [kw for kw in _KEYWORDS if kw in q]
    # 대표 키워드 1~2개
    key_part = " · ".join(hits[:2]) if hits else ""

    # 조립 규칙
    parts = [p for p in [dept_name, term_txt, key_part] if p]
    if parts:
        topic = " | ".join(parts) if (len(parts) >= 2) else parts[0]
        return _trim_topic(topic)

    # 마지막으로, 질문 전체에서 긴 수식어 제거한 핵심명사 후보
    # 불필요한 조사/마침표/따옴표 제거
    fallback = re.sub(r"[\"'“”‘’]", "", q)
    fallback = re.sub(r"(에 대해|만|으로|는|은|이|가|을|를|요)$", "", fallback)
    # 너무 길면 자름
    return _trim_topic(fallback) if fallback else ""

def _pick_intro(question: str, state: Dict[str, Any]) -> str:
    topic = _extract_topic(question, state)
    if topic:
        tpl = random.choice(INTRO_TEMPLATES_WITH_TOPIC)
        return tpl.format(topic=topic)
    return random.choice(INTRO_TEMPLATES_GENERIC)


# -------------------- 기존 유틸 --------------------

def _safe_path(h: Dict[str, Any]) -> str:
    return (h.get("path")
            or (h.get("metadata") or {}).get("path")
            or "").strip()

def _dedup_lines(text: str) -> str:
    out, prev = [], None
    for ln in (text or "").splitlines():
        cur = ln.strip()
        if cur and cur == prev:
            continue
        out.append(ln)
        prev = cur
    return "\n".join(out)

def _summarize_sources(hits: List[Dict[str, Any]], max_items: int = 8) -> str:
    if not hits:
        return "(없음)"
    lines = []
    for i, h in enumerate(hits[:max_items], 1):
        lines.append(f"{i}. {_safe_path(h)}")
    return "\n".join(lines)

def _build_context_from_hits(hits: List[Dict[str, Any]], *, max_items: int, budget_chars: int) -> str:
    if not hits:
        return ""
    # 연/학기 우선 정렬(메타가 있으면)
    def _ord(h):
        m = h.get("metadata") or {}
        y, s = m.get("year"), m.get("semester")
        return (0, *term_sort_key(y, s)) if (y or s) else (1, 99, 99)
    hits = sorted(hits, key=_ord)

    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits[:max_items], 1):
        path = _safe_path(h)
        body = _dedup_lines(h.get("document") or "")
        block = f"[SOURCE {i}] {path}\n{body}\n"
        blen = len(block)
        if used + blen > budget_chars:
            remain = budget_chars - used
            if remain > 200:  # 최소한의 컨텍스트 보장
                parts.append(block[:remain])
            break
        parts.append(block)
        used += blen
    return "\n\n---\n\n".join(parts)

def _make_llm(model_name: str, temperature: float, max_tokens: int):
    provider = config.llm_provider_from_model(model_name)
    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic 미설치 또는 로드 실패")
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
    else:
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai 미설치 또는 로드 실패")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)


# -------------------- 그래프 노드 --------------------

def node_parse_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state["question"]
    use_llm = bool(state["opts"].get("use_llm", True))
    hints_depts = state["opts"].get("scope_depts") or []

    if "micro_mode" not in state["opts"]:
        state["opts"]["micro_mode"] = None

    if not use_llm:
        state["context_struct"] = {"faculties": [], "departments": hints_depts, "year": None, "need_slots": []}
        return state

    try:
        model_name = state["opts"].get("model_name", config.LLM_MODEL)
        llm = _make_llm(model_name=model_name,
                        temperature=float(state["opts"].get("temperature") or 0.0),
                        max_tokens=int(state["opts"].get("max_tokens") or 400))  # 증가
        provider = config.llm_provider_from_model(model_name)
        if provider == "anthropic":
            sys = ("너는 대학 학사요람 Q&A용 추출기야. "
                   "입력 질문에서 단과대/학과/학년도/필요 슬롯을 JSON으로만 출력해. "
                   '반드시 {"faculties":[],"departments":[],"year":null,"need_slots":[]} 키를 사용.')
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                ("user", "질문: {q}\nJSON만 출력: faculties, departments, year, need_slots"),
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
                raise ValueError("parse_intent: JSON 파싱 실패(Claude)")
        else:
            sys = "너는 대학 학사요람 Q&A용 추출기다. 입력에서 단과대/학과/학년도/필요 슬롯을 구조화해라."
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                ("user", "질문: {q}\n출력은 JSON으로. keys: faculties, departments, year, need_slots"),
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
        state["context_struct"] = {"faculties": [], "departments": hints_depts, "year": None, "need_slots": []}
        return state

def node_need_more(state: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(state["opts"].get("use_llm", True)):
        state["needs_clarification"] = False
        return state
    ctx = state["context_struct"]
    if not ctx.get("departments") and not ctx.get("faculties"):
        state["needs_clarification"] = True
        state["clarification_prompt"] = "어느 학과(또는 전공) 기준인지 알려주세요. 예) 디지털미디어학과 / 소프트웨어학과"
    else:
        state["needs_clarification"] = False
    return state

def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("needs_clarification") or state.get("error") or state.get("skip_rag"):
        state["retrieved"] = []
        return state
    ctx = state["context_struct"]
    opts = state["opts"]
    try:
        hits = retrieve(
            state["question"],
            persist_dir=opts["persist_dir"], collection=opts["collection"], embedding_model=opts["embedding_model"],
            topk=int(opts.get("topk") or config.TOPK),
            lex_weight=float(opts.get("lex_weight") or config.LEX_WEIGHT),
            scope_colleges=(ctx.get("faculties") or None),
            scope_depts=(ctx.get("departments") or None),
            micro_mode=opts.get("micro_mode"),
            debug=bool(opts.get("debug") or False),
            rerank=bool(opts.get("rerank") or False),
            rerank_model=opts.get("rerank_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_candidates=int(opts.get("rerank_candidates") or 30),
            stitch_by_path=False,  # 섹션 확장 방식 사용
        )
        state["retrieved"] = hits
        return state
    except Exception as e:
        state["error"] = f"retrieval_error: {e}"
        state["retrieved"] = []
        return state

def node_build_context(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("error") or state.get("skip_rag"):
        state["context"] = ""
        return state
    hits = state.get("retrieved") or []
    opts = state["opts"]
    budget = int(opts.get("assemble_budget_chars") or 60000)  # 기본값 줄임
    max_items = int(opts.get("max_ctx_chunks") or 8)
    state["context"] = _build_context_from_hits(hits, max_items=max_items, budget_chars=budget)
    state["must_include"] = []
    return state

def _check_response_completeness(text: str) -> bool:
    """응답의 완성도를 체크"""
    text = (text or "").strip()
    if not text:
        return False
    # 한국어 문장 끝 패턴
    korean_endings = ('.', '요.', '다.', '니다.', '습니다.', '세요.', '어요.', '아요.', '지요.')
    # 출처 섹션이 있는 경우
    if '출처:' in text or '출처\n' in text:
        return True
    # 정상적인 문장 끝인 경우
    return text.endswith(korean_endings)

def node_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("skip_rag") or state.get("needs_clarification") or state.get("error"):
        return state

    hits = state.get("retrieved") or []
    use_llm = bool(state["opts"].get("use_llm", True))
    intro = _pick_intro(state.get("question", ""), state)

    # LLM 비사용 또는 히트 없음 → 요약+출처(인사말 포함)
    if not use_llm or not hits:
        src = _summarize_sources(hits)
        state["answer"] = f"{intro}\n\n검색된 문서 요약을 제공합니다.\n\n출처:\n{src}"
        state["llm_answer"] = None
        return state

    micro_mode = state["opts"].get("micro_mode", "exclude")
    style_guide = state.get("style_guide") or ""
    category = state.get("category") or "other"
    rule = {
        "exclude": "1) '마이크로전공' 내용은 제외하고 본전공 중심으로 답하세요.",
        "only": "1) 마이크로전공만 대상으로 답하세요.",
        "include": "1) 본전공과 마이크로 모두 포함하되, 본전공을 우선하세요.",
    }.get(micro_mode, "1) 본전공을 우선하되 필요 시 마이크로도 포함하세요.")

    # 간소화된 프롬프트
    persona = "아주대학교 학사안내 도우미로서 존댓말로 간결하게 답변하세요."
    sys = (f"{persona}\n"
           f"CONTEXT 근거로만 답하세요. 근거 없으면 '문서에서 확인되지 않습니다' 표기.\n"
           f"{rule}\n"
           f"가이드({category}): {style_guide}\n"
           "답변 후 반드시 '출처:' 섹션 추가.")

    # 컨텍스트 길이 체크 및 조정
    context = state['context']
    if len(context) > 50000:  # 너무 길면 줄임
        context = context[:45000] + "\n...(내용이 길어 일부 생략)..."

    # 인사말을 사용자 메시지 프롬프트에 포함 (LLM이 본문 앞에 두도록 유도)
    usr = f"{intro}\n\n질문: {state['question']}\n\nCONTEXT:\n{context}"

    # max_tokens 동적 조정
    base_max_tokens = int(state["opts"].get("max_tokens") or 1500)  # 기본값 증가
    if category == "term_plan":
        base_max_tokens = min(2500, base_max_tokens * 2)  # 학기별 계획은 더 긴 답변 필요

    llm = _make_llm(
        model_name=state["opts"].get("model_name", config.LLM_MODEL),
        temperature=float(state["opts"].get("temperature") or config.TEMPERATURE),
        max_tokens=base_max_tokens,
    )

    try:
        out = llm.invoke([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}])
        txt = (out.content or "").strip()

        # 응답 완성도 체크
        if not _check_response_completeness(txt):
            if not txt.endswith(("...", "…")):
                txt += "..."
            txt += "\n\n[응답이 일부 생략되었을 수 있습니다]"

        # 출처 추가(없다면)
        if "출처" not in txt and "SOURCE" not in txt:
            txt += "\n\n출처:\n" + _summarize_sources(hits)

        # 최종 세팅
        state["llm_answer"] = txt
        state["answer"] = txt
        return state

    except Exception as e:
        state["error"] = f"llm_error: {e}"
        state["answer"] = f"{intro}\n\n모델 호출에 실패했습니다.\n\n출처:\n" + _summarize_sources(hits)
        state["llm_answer"] = None
        return state