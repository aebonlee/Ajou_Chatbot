"""
앱의 HTTP 엔드포인트를 제공하는 FastAPI 서버 모듈.

구성 요약
- CORS/타이밍 로깅 등 공통 미들웨어
- 헬스체크(/health), 경량 메트릭(/metrics-lite)
- 핵심 RAG 엔드포인트: /yoram (요람/학과별)
- 보조 엔드포인트: /info(학사공통), /announcement(공지), /menu(임시)

설계 원칙
- '생성(LLM) 결과의 후처리(인사말/출처 병합)'는 그래프 레이어에서 수행.
  서버는 가급적 **추가 문자열을 붙이지 않고** 그래프 결과를 그대로 전달.
- 입력 검증(필수 파라미터)과 간단한 guard는 서버에서 처리.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid, time

from app.core import config
from app.core.config import CorpusType
from app.models.schemas import QueryRequest,NoticeQuery,NoticeResponse, InfoResponse, InfoRequest
from app.graphs.pipeline import run_rag_graph,rag_chain, route_query_sync # LangGraph 진입점
from app.utils.log import jlog, timed

# -----------------------------------------------------------------------------
# 앱/미들웨어
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Acad RAG API (LangChain+LangGraph)",
    default_response_class=ORJSONResponse,  # orjson 기반 더 빠른 JSON 응답
)

# CORS: 개발 초기에는 * 전체 허용, 배포 시 허용 도메인으로 제한 권장
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """
    모든 요청에 대해 처리 시간(ms)을 기록하는 경량 미들웨어.
    - 요청/응답 바디는 건드리지 않음
    - 실패 여부도 로깅
    """
    t0 = time.perf_counter()
    ok = True
    try:
        resp = await call_next(request)
        return resp
    except Exception:
        ok = False
        raise
    finally:
        try:
            jlog(
                span="http_request",
                route=request.url.path,
                ms=round((time.perf_counter() - t0) * 1000.0, 2),
                ok=ok,
            )
        except Exception:
            # 로깅 실패는 무시
            pass


# -----------------------------------------------------------------------------
# 헬스/메트릭
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """K8s/LB 헬스체크용 간단 엔드포인트."""
    return {"ok": True}

@app.get("/metrics-lite")
def metrics_lite():
    """추후 프로메테우스 익스포터 연동 전 간단 스텁."""
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# 공용 실행 유틸
# -----------------------------------------------------------------------------
def _run_graph(req: QueryRequest) -> dict:
    """
    LangGraph 파이프라인을 호출한다.
    - 서버는 파라미터를 **그대로 전달**하고,
    - 노드 계층에서 인사/출처 병합 등 후처리를 수행하도록 한다.
    """
    print(
        f"[API] q={req.question!r} depts={getattr(req, 'departments', None)} "
        f"collection={config.COLLECTION}"
    )

    return run_rag_graph(
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


# -----------------------------------------------------------------------------
# 보조 스키마 (엔드포인트별 입력 전용)
# -----------------------------------------------------------------------------
# class InfoRequest(BaseModel):
#     """학사 공통(학칙/학사력 등) 조회용 요청 바디."""
#     question: str
#     topics: List[str] = Field(default_factory=list)
#     topk: int = 8
#     debug: bool = False
#     use_llm: bool = True
#     micro_mode: str = "exclude"
#     assemble_budget_chars: Optional[int] = None
#     max_ctx_chunks: Optional[int] = None
#     rerank: Optional[bool] = None
#     rerank_model: Optional[str] = None
#     rerank_candidates: Optional[int] = None

class AnnouncementRequest(BaseModel):
    """학과/단과대 공지 조회용 요청 바디."""
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
    """식단 조회 임시 스키마(추후 실제 소스 연동 예정)."""
    question: Optional[str] = ""


# -----------------------------------------------------------------------------
# 핵심 엔드포인트: /yoram (요람/학과별)
# -----------------------------------------------------------------------------
@app.post("/yoram")
async def post_yoram(req: QueryRequest, request: Request):
    """
    학과별(요람) 질문을 받아 RAG 그래프를 실행하고 결과를 그대로 반환한다.
    - 서버는 인사/출처를 추가하지 않는다(그래프 레이어가 책임짐).
    """
    rid = str(uuid.uuid4())

    try:
        # 요청 로깅(선택)
        try:
            jlog(
                event="request", route="/yoram", request_id=rid,
                question=req.question, depts=getattr(req, "departments", None),
                opts=req.dict(),
            )
        except Exception:
            pass

        out = _run_graph(req)

        # 결과 로깅(선택)
        try:
            jlog(
                event="result", route="/yoram", request_id=rid,
                error=out.get("error"), sources=len(out.get("sources") or []),
            )
        except Exception:
            pass

        # 그래프가 생성한 결과를 **있는 그대로** 전달
        return {
            "question": out.get("question") or req.question,
            "answer": out.get("answer"),
            "llm_answer": out.get("llm_answer"),
            "context": out.get("context"),
            "sources": out.get("sources") or [],
            "micro_mode": out.get("micro_mode", "exclude"),
            "error": out.get("error"),
            "clarification": out.get("clarification_prompt"),
        }

    except Exception as e:
        # 서버 레벨 예외는 공통 메시지로 대응
        try:
            jlog(event="exception", route="/yoram", request_id=rid, error=str(e))
        except Exception:
            pass
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


# # -----------------------------------------------------------------------------
# # 보조 엔드포인트: /info (학사공통)
# # -----------------------------------------------------------------------------
# @app.post("/info")
# async def post_info(req: InfoRequest, request: Request):
#     """
#     학사 공통 범주(topics)에 대한 질문.
#     - 현재는 동일 그래프를 재사용하며, topics를 질문 앞에 힌트로 부착.
#       (필요하면 별도 컬렉션/그래프 분리 가능)
#     """
#     rid = str(uuid.uuid4())
#
#     if not req.topics:
#         return {
#             "question": req.question,
#             "answer": "제게 물어볼 범주(학칙/학사력/대학생활안내 등) 중 최소 1개를 선택해 주세요.",
#             "llm_answer": None,
#             "context": "",
#             "sources": [],
#             "micro_mode": "exclude",
#             "error": "bad_request: no_topics",
#             "clarification": None,
#         }
#
#     try:
#         try:
#             jlog(event="request", route="/info", request_id=rid,
#                  question=req.question, topics=req.topics)
#         except Exception:
#             pass
#
#         out = run_rag_graph(
#             question=f"[{', '.join(req.topics)}] {req.question}",
#             persist_dir=config.PERSIST_DIR,
#             collection=config.COLLECTION,            # 필요 시 분리
#             embedding_model=config.EMBEDDING_MODEL,
#             topk=req.topk,
#             model_name=config.LLM_MODEL,
#             temperature=config.TEMPERATURE,
#             max_tokens=config.MAX_TOKENS,
#             use_llm=req.use_llm,
#             debug=req.debug,
#             scope_depts=None,
#             micro_mode=req.micro_mode,
#             assemble_budget_chars=req.assemble_budget_chars,
#             max_ctx_chunks=req.max_ctx_chunks,
#             rerank=req.rerank or False,
#             rerank_model=req.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
#             rerank_candidates=req.rerank_candidates or 30,
#         )
#
#         try:
#             jlog(event="result", route="/info", request_id=rid,
#                  error=out.get("error"), sources=len(out.get("sources") or []))
#         except Exception:
#             pass
#
#         return {
#             "question": out.get("question") or req.question,
#             "answer": out.get("answer"),
#             "llm_answer": out.get("llm_answer"),
#             "context": out.get("context"),
#             "sources": out.get("sources") or [],
#             "micro_mode": out.get("micro_mode", "exclude"),
#             "error": out.get("error"),
#             "clarification": out.get("clarification_prompt"),
#         }
#
#     except Exception as e:
#         return {
#             "question": req.question,
#             "answer": "요청 처리 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요.",
#             "llm_answer": None,
#             "context": "",
#             "sources": [],
#             "micro_mode": "exclude",
#             "error": f"server_error: {e}",
#             "clarification": None,
#         }


# -----------------------------------------------------------------------------
# 보조 엔드포인트: /notice (공지)
# -----------------------------------------------------------------------------


@app.post("/notice", response_model=NoticeResponse)
async def get_notice_answer(query: NoticeQuery):
    """
    사용자의 질문에 대한 공지사항을 검색하고 답변합니다.
    """
    try:
        # 2. rag_chain 호출
        answer = rag_chain.invoke({"question": query.question})
        return NoticeResponse(answer=answer)
    except Exception as e:
        # 오류 처리
        return {"answer": f"오류가 발생했습니다: {str(e)}"}

@app.post("/announcement", response_model=NoticeResponse)
# async def post_announcement_bridge(query: NoticeQuery):
#     return await get_notice_answer(query)

async def post_announcement(req: AnnouncementRequest):  # AnnouncementRequest 사용
    """
    학과 정보를 포함한 공지사항 검색
    """
    print(f"[ANNOUNCEMENT] ===== 요청 받음 =====")
    print(f"[ANNOUNCEMENT] Question: {req.question}")
    print(f"[ANNOUNCEMENT] Departments: {req.departments}")

    try:
        from app.services.retriever import get_enhanced_filter, dynamic_retriever

        # 질문 강화
        enhanced_question = req.question
        if req.departments:
            dept_hint = " ".join(req.departments)
            enhanced_question = f"{dept_hint} {enhanced_question}"

        print(f"[ANNOUNCEMENT] Enhanced: {enhanced_question}")

        # 필터 생성
        filter_dict = get_enhanced_filter(enhanced_question)
        print(f"[ANNOUNCEMENT] Filter: {filter_dict}")

        # 검색 실행
        docs = dynamic_retriever(enhanced_question, filter_dict)
        print(f"[ANNOUNCEMENT] Found docs: {len(docs)}")

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            print(f"[ANNOUNCEMENT] Doc {i}: {meta.get('title')} | {meta.get('department_name')}")

        if not docs:
            answer = "관련 공지사항을 찾을 수 없습니다."
        else:
            # 직접 포맷
            lines = [f"총 {len(docs)}개의 공지사항을 찾았습니다:\n"]
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                lines.append(f"{i}. {meta.get('title', 'N/A')}")
                lines.append(f"   URL: {meta.get('url', 'N/A')}")
                lines.append("")
            answer = "\n".join(lines)

        print(f"[ANNOUNCEMENT] Response length: {len(answer)}")
        return NoticeResponse(answer=answer)

    except Exception as e:
        print(f"[ANNOUNCEMENT] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return NoticeResponse(answer=f"오류 발생: {str(e)}")


# -----------------------------------------------------------------------------
# 임시 엔드포인트: /menu (식단)
# -----------------------------------------------------------------------------
@app.post("/menu")
async def post_menu(req: MenuRequest, request: Request):
    """
    식단 정보는 아직 별도 소스/크롤러 미연동이므로 임시 응답을 제공한다.
    """
    q = (req.question or "").strip()
    body = "오늘의 식단 정보는 준비 중입니다. 캠퍼스/식당명을 알려주시면 더 정확히 안내할게요."
    return {
        "question": q,
        "answer": body,        # 그래프와 동일 포맷으로 간단 본문만 반환
        "llm_answer": body,
        "context": "",
        "sources": [],
        "micro_mode": "exclude",
        "error": None,
        "clarification": None,
    }


# -----------------------------------------------------------------------------
# 학사공통 : /info
# -----------------------------------------------------------------------------


@app.post("/info", response_model=InfoResponse)
@timed("http_post_info")
def info_query(req: InfoRequest, request: Request):
    request_id = str(uuid.uuid4())
    try:
        jlog(event="request", route="/info", request_id=request_id,
             question=req.question, selected_list=req.selected_list, opts=req.dict())

        # selected_list 유효성 검사
        valid_departments = [ctype.value for ctype in CorpusType]
        for dept in req.selected_list:
            if dept not in valid_departments:
                raise HTTPException(status_code=400, detail=f"Invalid department: {dept}. Must be one of {valid_departments}")

        result = route_query_sync(
            question=req.question,
            departments=req.selected_list
        )

        jlog(event="result", route="/info", request_id=request_id,
             question=req.question, departments=req.selected_list, success=True)

        return InfoResponse(
            answer=result["answer"],
            question=req.question
        )

    except HTTPException:
        raise
    except Exception as e:
        jlog(event="exception", route="/info", request_id=request_id,
             question=req.question, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")