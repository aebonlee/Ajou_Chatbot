# app/api/server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from app.core import config
from app.models.schemas import QueryRequest  # /yoram에서 사용
from app.graphs.pipeline import run_rag_graph
from app.utils.log import jlog

import uuid
import time


# =========================
# FastAPI 앱 & 미들웨어
# =========================
app = FastAPI(
    title="Acad RAG API (LangChain+LangGraph)",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 허용 도메인만 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    ok = True
    try:
        response = await call_next(request)
        return response
    except Exception:
        ok = False
        raise
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        route = request.url.path
        try:
            jlog(span="http_request", route=route, ms=round(dt_ms, 2), ok=ok)
        except Exception:
            pass


# =========================
# 유틸
# =========================
def _intro_line(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "문의해 주셔서 고마워요! 아래에 핵심만 정리해 드릴게요."
    return f"“{q}”에 대해 질문해 주셨군요! 아래에 핵심만 정리해 드릴게요."

def _append_sources(answer_core: str, sources):
    srcs = sources or []
    if not srcs:
        return answer_core
    lines = [str(s).strip() for s in srcs if str(s).strip()]
    uniq, seen = [], set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    src_block = "\n".join(f"- {s}" for s in uniq)
    return f"{answer_core}\n\n출처:\n{src_block}"

def _run_graph(req: QueryRequest):
    # 디버그 로그
    print(f"API DEBUG - Question: {req.question}")
    print(f"API DEBUG - Departments: {getattr(req, 'departments', None)}")
    print(f"API DEBUG - Config Collection: {config.COLLECTION}")
    print(f"API DEBUG - Debug: {req.debug}")

    result = run_rag_graph(
        question=req.question,
        persist_dir=config.PERSIST_DIR,
        collection=config.COLLECTION,
        embedding_model=config.EMBEDDING_MODEL,
        topk=req.topk,
        model_name=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        use_llm=req.use_llm,
        debug=req.debug,
        scope_depts=(getattr(req, "departments", None) or None),
        micro_mode=req.micro_mode,
        assemble_budget_chars=req.assemble_budget_chars,
        max_ctx_chunks=req.max_ctx_chunks,
        rerank=req.rerank or False,
        rerank_model=req.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates=req.rerank_candidates or 30,
    )

    ctx = result.get("context") or ""
    print(f"API DEBUG - Result context length: {len(ctx)}")
    print("===== CONTEXT START =====")
    print(ctx)
    print("===== CONTEXT END =====")
    return result

def _build_final_answer(out: dict, original_q: str) -> str:
    intro = _intro_line(out.get("question") or original_q or "")
    core = (out.get("answer") or out.get("llm_answer") or "").strip()
    friendly = f"{intro}\n\n{core}" if core else intro
    return _append_sources(friendly, out.get("sources") or [])


# =========================
# 헬스체크/메트릭
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics-lite")
def metrics_lite():
    return {"status": "ok"}


# =========================
# 스키마 (엔드포인트별 요청 바디)
# =========================
class InfoRequest(BaseModel):
    question: str
    topics: List[str] = Field(default_factory=list)
    topk: int = 8
    debug: bool = False
    use_llm: bool = True
    micro_mode: str = "exclude"
    assemble_budget_chars: Optional[int] = None
    max_ctx_chunks: Optional[int] = None
    rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_candidates: Optional[int] = None

class AnnouncementRequest(BaseModel):
    question: str
    departments: List[str] = Field(default_factory=list)
    topk: int = 8
    debug: bool = False
    use_llm: bool = True
    micro_mode: str = "exclude"
    assemble_budget_chars: Optional[int] = None
    max_ctx_chunks: Optional[int] = None
    rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_candidates: Optional[int] = None

class MenuRequest(BaseModel):
    question: Optional[str] = ""


# =========================
# /yoram : 요람(학과별)
# =========================
@app.post("/yoram")
async def post_yoram(req: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())
    try:
        try:
            jlog(
                event="request", route="/yoram", request_id=request_id,
                question=req.question, depts=getattr(req, "departments", None), opts=req.dict()
            )
        except Exception:
            pass

        out = _run_graph(req)

        try:
            jlog(
                event="result", route="/yoram", request_id=request_id,
                error=out.get("error"), sources=len(out.get("sources") or [])
            )
        except Exception:
            pass

        final_answer = _build_final_answer(out, req.question)

        return {
            "question": out.get("question"),
            "answer": final_answer,                         # 최종(인사말+본문+출처)
            "llm_answer": (out.get("answer") or out.get("llm_answer") or "").strip() or None,  # 본문만
            "context": out.get("context"),
            "sources": out.get("sources") or [],
            "micro_mode": out.get("micro_mode", "exclude"),
            "error": out.get("error"),
            "clarification": out.get("clarification_prompt"),
        }

    except Exception as e:
        print(f"API DEBUG - Exception: {e}")
        try:
            jlog(event="exception", route="/yoram", request_id=request_id, error=str(e))
        except Exception:
            pass
        return {
            "question": getattr(req, "question", None),
            "answer": "요청 처리 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요.",
            "llm_answer": None,
            "context": "",
            "sources": [],
            "micro_mode": "exclude",
            "error": f"server_error: {e}",
            "clarification": None,
        }


# =========================
# /info : 학사공통(학칙/학사력/대학생활안내 등)
# =========================
@app.post("/info")
async def post_info(req: InfoRequest, request: Request):
    request_id = str(uuid.uuid4())

    if not req.topics:
        return {
            "question": req.question,
            "answer": "제게 물어볼 범주(학칙/학사력/대학생활안내 등) 중 최소 1개를 선택해 주세요.",
            "llm_answer": None,
            "context": "",
            "sources": [],
            "micro_mode": "exclude",
            "error": "bad_request: no_topics",
            "clarification": None,
        }

    try:
        try:
            jlog(event="request", route="/info", request_id=request_id,
                 question=req.question, topics=req.topics)
        except Exception:
            pass

        # 간단히 토픽을 프롬프트에 주입하여 같은 그래프 재사용
        out = run_rag_graph(
            question=f"[{', '.join(req.topics)}] {req.question}",
            persist_dir=config.PERSIST_DIR,
            collection=config.COLLECTION,               # 필요시 별도 컬렉션 분리 가능
            embedding_model=config.EMBEDDING_MODEL,
            topk=req.topk,
            model_name=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            use_llm=req.use_llm,
            debug=req.debug,
            scope_depts=None,
            micro_mode=req.micro_mode,
            assemble_budget_chars=req.assemble_budget_chars,
            max_ctx_chunks=req.max_ctx_chunks,
            rerank=req.rerank or False,
            rerank_model=req.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_candidates=req.rerank_candidates or 30,
        )

        try:
            jlog(event="result", route="/info", request_id=request_id,
                 error=out.get("error"), sources=len(out.get("sources") or []))
        except Exception:
            pass

        final_answer = _build_final_answer(out, req.question)
        return {
            "question": out.get("question") or req.question,
            "answer": final_answer,
            "llm_answer": (out.get("answer") or out.get("llm_answer") or "").strip() or None,
            "context": out.get("context"),
            "sources": out.get("sources") or [],
            "micro_mode": out.get("micro_mode", "exclude"),
            "error": out.get("error"),
            "clarification": out.get("clarification_prompt"),
        }

    except Exception as e:
        return {
            "question": req.question,
            "answer": "요청 처리 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요.",
            "llm_answer": None,
            "context": "",
            "sources": [],
            "micro_mode": "exclude",
            "error": f"server_error: {e}",
            "clarification": None,
        }


# =========================
# /announcement : 공지사항
# =========================
@app.post("/announcement")
async def post_announcement(req: AnnouncementRequest, request: Request):
    request_id = str(uuid.uuid4())

    if not req.departments:
        return {
            "question": req.question,
            "answer": "어느 학과(또는 단과대) 공지인지 최소 1개를 선택해 주세요.",
            "llm_answer": None,
            "context": "",
            "sources": [],
            "micro_mode": "exclude",
            "error": "bad_request: no_departments",
            "clarification": None,
        }

    try:
        try:
            jlog(event="request", route="/announcement", request_id=request_id,
                 question=req.question, depts=req.departments)
        except Exception:
            pass

        out = run_rag_graph(
            question=req.question,
            persist_dir=config.PERSIST_DIR,
            collection=config.COLLECTION,         # 공지별 분리 시 교체
            embedding_model=config.EMBEDDING_MODEL,
            topk=req.topk,
            model_name=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            use_llm=req.use_llm,
            debug=req.debug,
            scope_depts=req.departments,
            micro_mode=req.micro_mode,
            assemble_budget_chars=req.assemble_budget_chars,
            max_ctx_chunks=req.max_ctx_chunks,
            rerank=req.rerank or False,
            rerank_model=req.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_candidates=req.rerank_candidates or 30,
        )

        try:
            jlog(event="result", route="/announcement", request_id=request_id,
                 error=out.get("error"), sources=len(out.get("sources") or []))
        except Exception:
            pass

        final_answer = _build_final_answer(out, req.question)
        return {
            "question": out.get("question") or req.question,
            "answer": final_answer,
            "llm_answer": (out.get("answer") or out.get("llm_answer") or "").strip() or None,
            "context": out.get("context"),
            "sources": out.get("sources") or [],
            "micro_mode": out.get("micro_mode", "exclude"),
            "error": out.get("error"),
            "clarification": out.get("clarification_prompt"),
        }

    except Exception as e:
        return {
            "question": req.question,
            "answer": "요청 처리 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요.",
            "llm_answer": None,
            "context": "",
            "sources": [],
            "micro_mode": "exclude",
            "error": f"server_error: {e}",
            "clarification": None,
        }


# =========================
# /menu : 식단 (임시 플레이스홀더)
# =========================
@app.post("/menu")
async def post_menu(req: MenuRequest, request: Request):
    q = (req.question or "").strip()
    base = "오늘의 식단 정보는 준비 중입니다. 최신 식단은 학교 식단 안내 페이지를 참고해 주세요."
    tip = 'TIP: 어떤 캠퍼스/식당인지 알려주시면 더 정확히 안내할게요.'
    answer_core = f"{base}\n\n{tip}"
    final_answer = _append_sources(f"{_intro_line(q)}\n\n{answer_core}", [])
    return {
        "question": q,
        "answer": final_answer,
        "llm_answer": answer_core,
        "context": "",
        "sources": [],
        "micro_mode": "exclude",
        "error": None,
        "clarification": None,
    }