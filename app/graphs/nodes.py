# app/graphs/nodes.py
"""
LangGraph 노드 정의:
- parse_intent: (LLM 우선) 단과대/학과 추출 + 마이크로 모드 결정
- need_more: 학과 미지정 시 clarification 프롬프트 반환
- retrieve: 스코프/마이크로 모드 반영하여 하이브리드 검색 (+옵션: 재랭크)
- build_context: 예산 기반으로 CONTEXT 문자열 생성 (심플 스티칭)
- answer: LLM 생성 (CONTEXT 기반, '출처' 포함) — 단순 1회 생성, 재시도/검증 없음
"""
from typing import List, Dict, Any
import json
import re

# 두 프로바이더를 모두 지원 (설치 안되어 있으면 None)
try:
    from langchain_openai import ChatOpenAI  # OpenAI
except Exception:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic  # Anthropic
except Exception:
    ChatAnthropic = None  # type: ignore

from langchain_core.prompts import ChatPromptTemplate

from app.core import config
from app.models.schemas import QuerySchema
from app.services.retriever import retrieve  # ⬅️ detect_micro_mode 자동감지 제거

# ---------- 유틸 ----------
def _safe_path(h: Dict[str, Any]) -> str:
    """hit에서 path를 안전하게 얻는다."""
    return (h.get("path")
            or (h.get("metadata") or {}).get("path")
            or "").strip()

def _body_without_path_prefix(doc: str) -> str:
    """문서가 '[PATH] ...'로 시작하면 그 1행을 제거하고 본문만 반환."""
    if not doc:
        return ""
    if doc.startswith("[PATH]") and "\n" in doc:
        return doc.split("\n", 1)[1].strip()
    return doc.strip()

def _dedup_lines(text: str) -> str:
    """연속 중복 라인 제거(LLM 혼동 완화)."""
    out, prev = [], None
    for ln in text.splitlines():
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

def _build_context_from_hits(
    hits: List[Dict[str, Any]],
    *,
    max_items: int,
    budget_chars: int,
) -> str:
    """
    예산(budget_chars)을 넘지 않는 선에서 상위 hit들을 순서대로 붙인다.
    - 각 블록은 [SOURCE i] <path> + 본문
    """
    if not hits:
        return ""

    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits[:max_items], 1):
        path = _safe_path(h)
        body = _dedup_lines(_body_without_path_prefix(h.get("document") or ""))
        block = f"[SOURCE {i}] {path}\n{body}\n"
        blen = len(block)
        if used + blen > budget_chars:
            remain = budget_chars - used
            if remain > 0:
                parts.append(block[:remain])
            break
        parts.append(block)
        used += blen

    return "\n\n---\n\n".join(parts)

# ---------- LLM 팩토리 ----------
def _make_llm(model_name: str, temperature: float, max_tokens: int):
    """
    모델명으로 프로바이더(OpenAI/Anthropic)를 자동 판별하여 LLM 인스턴스 생성.
    - 'claude-*' → ChatAnthropic
    - 그 외 → ChatOpenAI
    """
    provider = config.llm_provider_from_model(model_name)
    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic 미설치 또는 로드 실패")
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
    else:
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai 미설치 또는 로드 실패")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)

# ---------- 노드 ----------
def node_parse_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state["question"]
    use_llm = bool(state["opts"].get("use_llm", True))
    # 프론트에서 scope_depts로 넘겨준 학과 힌트
    hints_depts = state["opts"].get("scope_depts") or []

    # ⬇️ 자동 마이크로 감지 제거: 프론트가 명시 전달한 값만 유지
    if "micro_mode" not in state["opts"]:
        state["opts"]["micro_mode"] = None  # retriever가 자동 판정할 수 있게 None 유지

    if not use_llm:
        # LLM 미사용 시 힌트만 반영
        state["context_struct"] = {
            "faculties": [],
            "departments": hints_depts,
            "year": None,
            "need_slots": [],
        }
        return state

    # LLM 구조화 파싱 (Claude/OpenAI 모두 지원)
    try:
        model_name = state["opts"].get("model_name", config.LLM_MODEL)
        llm = _make_llm(
            model_name=model_name,
            temperature=float(state["opts"].get("temperature") or 0.0),
            max_tokens=int(state["opts"].get("max_tokens") or 300),
        )

        provider = config.llm_provider_from_model(model_name)
        if provider == "anthropic":
            # Claude: JSON으로만 출력하게 유도 + 폴백 파서
            sys = (
                "너는 대학 학사요람 Q&A용 추출기야. "
                "입력 질문에서 단과대/학과/학년도/필요 슬롯을 JSON으로만 출력해. "
                '반드시 {"faculties":[],"departments":[],"year":null,"need_slots":[]} 키를 사용.'
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                ("user", "질문: {q}\nJSON만 출력: faculties, departments, year, need_slots"),
            ])
            msg = prompt.format_messages(q=q)
            out = llm.invoke(msg)
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
            # OpenAI: LangChain structured_output 사용
            sys = "너는 대학 학사요람 Q&A용 추출기다. 입력에서 단과대/학과/학년도/필요 슬롯을 구조화해라."
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys),
                ("user", "질문: {q}\n출력은 JSON으로. keys: faculties, departments, year, need_slots"),
            ])
            out = llm.with_structured_output(QuerySchema).invoke(prompt.format_messages(q=q))
            qs = out  # 이미 QuerySchema

        depts_union = list({*(qs.departments or []), *hints_depts})
        state["context_struct"] = {
            "faculties": list(qs.faculties or []),
            "departments": depts_union,
            "year": qs.year,
            "need_slots": list(qs.need_slots or []),
        }
        return state

    except Exception:
        state["context_struct"] = {
            "faculties": [],
            "departments": hints_depts,
            "year": None,
            "need_slots": [],
        }
        return state


def node_need_more(state: Dict[str, Any]) -> Dict[str, Any]:
    # 검색-only 모드에선 clarification 미사용
    if not bool(state["opts"].get("use_llm", True)):
        state["needs_clarification"] = False
        return state

    ctx = state["context_struct"]
    # 학과/전공 힌트가 전혀 없으면 되물어보기
    if not ctx.get("departments") and not ctx.get("faculties"):
        state["needs_clarification"] = True
        state["clarification_prompt"] = "어느 학과(또는 전공) 기준인지 알려주세요. 예) 디지털미디어학과 / 소프트웨어학과"
    else:
        state["needs_clarification"] = False
    return state


def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("needs_clarification") or state.get("error"):
        state["retrieved"] = []
        return state
    if state.get("skip_rag"):
        state["retrieved"] = []
        return state

    ctx = state["context_struct"]
    opts = state["opts"]
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
            # 전역/옵션 기반으로 스티칭 여부 결정 (기본 False)
            stitch_by_path=bool(opts.get("stitch_by_path") if "stitch_by_path" in opts else config.STITCH_BY_PATH),
        )
        state["retrieved"] = hits
        return state
    except Exception as e:
        state["error"] = f"retrieval_error: {e}"
        state["retrieved"] = []
        return state


def node_build_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """INDEX/강제 포함 힌트 없이 스티칭만."""
    if state.get("error"):
        state["context"] = ""
        return state
    if state.get("skip_rag"):
        state["context"] = ""
        return state

    hits = state.get("retrieved") or []
    opts = state["opts"]
    budget = int(opts.get("assemble_budget_chars") or 40000)
    max_items = int(opts.get("max_ctx_chunks") or 12)

    stitched = _build_context_from_hits(
        hits,
        max_items=max_items,
        budget_chars=budget,
    )
    state["context"] = stitched
    # 참고: 예전 모드에서는 must_include 사용 안 함
    state["must_include"] = []
    return state


def node_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """단순 프롬프트 → 1회 생성 → 그대로 llm_answer/answer."""
    # 고정응답이면 그대로 반환
    if state.get("skip_rag"):
        return state
    # clarification or error 시 생성 생략
    if state.get("needs_clarification") or state.get("error"):
        return state

    hits = state.get("retrieved") or []
    use_llm = bool(state["opts"].get("use_llm", True))
    if not use_llm or not hits:
        # 폴백/미사용: 간단 요약 + 출처 나열
        src = _summarize_sources(hits)
        state["answer"] = "검색된 문서 요약을 제공합니다.\n\n출처:\n" + src
        state["llm_answer"] = None
        return state

    micro_mode = state["opts"].get("micro_mode", "exclude")
    style_guide = state.get("style_guide") or ""
    category = state.get("category") or "other"

    # 카테고리/모드별 규칙
    if category == "micro_list":
        rule = (
            "1) 마이크로전공만 대상으로 답하세요.\n"
            "2) 전공명 목록과 각 1문장 특징만 적고, 과목 긴 나열/학점표/학기별 표는 쓰지 마세요.\n"
            "3) 중복 문장/표기는 제거하세요.\n"
        )
    elif category == "micro_detail":
        rule = (
            "1) 마이크로전공만 대상으로 답하세요.\n"
            "2) 이수학점(총/필수), 핵심 필수 1과목, 특징 1줄만 요약하세요.\n"
            "3) 불필요한 과목 나열은 피하세요.\n"
        )
    else:
        rule = {
            "exclude": "1) '마이크로전공' 내용은 제외하고 본전공 중심으로 답하세요.",
            "only":    "1) 마이크로전공만 대상으로 답하세요.",
            "include": "1) 본전공과 마이크로 모두 포함하되, 본전공을 우선하세요.",
        }.get(micro_mode, "1) 본전공을 우선하되 필요 시 마이크로도 포함하세요.")

    persona = (
        "페르소나: 당신은 아주대학교 신입생을 돕는 친근한 학사안내 도우미입니다. "
        "존댓말로 간결하고 또렷하게 설명하세요."
    )
    sys = (
        f"{persona}\n"
        "반드시 CONTEXT 근거에서만 답하세요. 근거가 없으면 '문서에서 확인되지 않습니다'를 포함하세요.\n"
        f"{rule}\n"
        f"스타일 가이드({category}): {style_guide}\n"
        "가능하면 불릿으로 정리하고, 마지막에 '출처' 석션에 SOURCE 경로를 요약하세요."
    )
    usr = f"질문: {state['question']}\n\nCONTEXT:\n{state['context']}\n\n출처는 '출처:' 섹션으로 끝에 표기."

    # LLM 인스턴스
    llm = _make_llm(
        model_name=state["opts"].get("model_name", config.LLM_MODEL),
        temperature=float(state["opts"].get("temperature") or config.TEMPERATURE),
        max_tokens=int(state["opts"].get("max_tokens") or config.MAX_TOKENS),
    )

    try:
        out = llm.invoke([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}])
        txt = (out.content or "").strip()
        if "출처" not in txt:
            txt += "\n\n출처:\n" + _summarize_sources(hits)

        # 한 번 생성한 텍스트를 그대로 노출
        state["llm_answer"] = txt
        state["answer"] = txt
        return state

    except Exception as e:
        state["error"] = f"llm_error: {e}"
        state["answer"] = "모델 호출에 실패했습니다.\n\n출처:\n" + _summarize_sources(hits)
        state["llm_answer"] = None
        return state