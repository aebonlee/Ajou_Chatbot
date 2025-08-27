from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from app.core import config
from app.models.schemas import QueryRequest  # QueryResponse는 사용하지 않음
from app.graphs.pipeline import run_rag_graph
from app.utils.log import jlog
import uuid
import time

# 기본 응답 클래스를 ORJSON으로 통일
app = FastAPI(
    title="Acad RAG API (LangChain+LangGraph)",
    default_response_class=ORJSONResponse,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 간단 타이밍 미들웨어 (바디는 건드리지 않음) ---
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
        # path 로깅
        route = request.url.path
        try:
            jlog(span="http_request", route=route, ms=round(dt_ms, 2), ok=ok)
        except Exception:
            # 로깅 실패해도 요청 처리는 유지
            pass


@app.get("/health")
def health():
    return {"ok": True}


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
    uniq = []
    seen = set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    src_block = "\n".join(f"- {s}" for s in uniq)
    return f"{answer_core}\n\n출처:\n{src_block}"


def _run_graph(req: QueryRequest):
    # 디버그 로그 (stdout)
    print(f"API DEBUG - Question: {req.question}")
    print(f"API DEBUG - Departments: {req.departments}")
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
        scope_depts=(req.departments or None),
        micro_mode=req.micro_mode,
        assemble_budget_chars=req.assemble_budget_chars,
        max_ctx_chunks=req.max_ctx_chunks,
        rerank=req.rerank or False,
        rerank_model=req.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates=req.rerank_candidates or 30,
    )

    # 디버그: run_rag_graph 결과 확인
    ctx = result.get("context") or ""
    print(f"API DEBUG - Result context length: {len(ctx)}")
    print("===== CONTEXT START =====")
    print(ctx)
    print("===== CONTEXT END =====")
    return result


@app.post("/yoram")  # response_model 제거(직접 dict 반환)
async def post_query(req: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())

    try:
        # 요청 로그
        try:
            jlog(
                event="request", route="/yoram", request_id=request_id,
                question=req.question, depts=req.departments, opts=req.dict()
            )
        except Exception:
            pass  # 로깅 실패 무시

        out = _run_graph(req)

        # 결과 로그
        try:
            jlog(
                event="result", route="/yoram", request_id=request_id,
                error=out.get("error"), sources=len(out.get("sources") or [])
            )
        except Exception:
            pass

        # LLM이 만든 '핵심 본문'에 인사 + 출처 붙여 최종 answer 구성
        intro = _intro_line(out.get("question") or req.question or "")
        core = out.get("answer") or out.get("llm_answer") or ""
        core = core.strip()
        friendly = f"{intro}\n\n{core}" if core else intro
        final_answer = _append_sources(friendly, out.get("sources") or [])

        # 응답 dict (ORJSONResponse가 자동 직렬화)
        response_dict = {
            "question": out.get("question"),
            "answer": final_answer,            # ✅ 사용자에게 보여줄 최종 응답
            "llm_answer": core or None,        # (옵션) 인사/출처 제거된 핵심 본문만
            "context": out.get("context"),
            "sources": out.get("sources") or [],
            "micro_mode": out.get("micro_mode", "exclude"),
            "error": out.get("error"),
            "clarification": out.get("clarification_prompt"),
        }

        # 디버그: 응답 데이터 길이
        print(f"API DEBUG - Response context length: {len(response_dict.get('context') or '')}")
        print(f"API DEBUG - Response answer length: {len(response_dict.get('answer') or '')}")
        print(f"API DEBUG - Response sources count: {len(response_dict.get('sources') or [])}")

        return response_dict

    except Exception as e:
        print(f"API DEBUG - Exception: {e}")
        try:
            jlog(event="exception", route="/yoram", request_id=request_id, error=str(e))
        except Exception:
            pass

        # 에러도 dict로 반환
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


@app.get("/metrics-lite")
def metrics_lite():
    return {"status": "ok"}  # 운영용 exporter 붙일 예정이면 교체