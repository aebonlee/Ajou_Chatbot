from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core import config
from app.models.schemas import QueryRequest, QueryResponse
from app.graphs.pipeline import run_rag_graph
from app.utils.log import jlog, timed
import uuid

app = FastAPI(title="Acad RAG API (LangChain+LangGraph)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def _run_graph(req: QueryRequest):
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
        scope_depts=(req.departments or None),
        micro_mode=req.micro_mode,
        assemble_budget_chars=req.assemble_budget_chars,
        max_ctx_chunks=req.max_ctx_chunks,
        rerank=req.rerank or False,
        rerank_model=req.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates=req.rerank_candidates or 30,
    )

@app.post("/yoram", response_model=QueryResponse)
@timed("http_post_yoram")
def post_query(req: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())
    try:
        jlog(event="request", route="/yoram", request_id=request_id,
             question=req.question, depts=req.departments, opts=req.dict())
        out = _run_graph(req)
        jlog(event="result", route="/yoram", request_id=request_id,
             error=out.get("error"), sources=len(out.get("sources", [])))
        return QueryResponse(
            question=out.get("question"),
            answer=out.get("answer"),
            llm_answer=out.get("llm_answer"),
            context=out.get("context"),
            sources=out.get("sources", []),
            micro_mode=out.get("micro_mode", "exclude"),
            error=out.get("error"),
            clarification=out.get("clarification_prompt"),
        )
    except Exception as e:
        jlog(event="exception", route="/yoram", request_id=request_id, error=str(e))
        return QueryResponse(
            question=req.question,
            answer=None,
            llm_answer=None,
            context="",
            sources=[],
            micro_mode=out.get("micro_mode", "exclude") if 'out' in locals() else "exclude",
            error=f"server_error: {e}",
            clarification=None,
        )

# (선택) 운영 확인용 간이 메트릭
@app.get("/metrics-lite")
def metrics_lite():
    return {"status":"ok"}  # Prometheus를 붙일 계획이면 여기서 exporter 사용