# ... 상단 import 동일
from app.core import config
from app.models.schemas import QuerySchema
from app.services.retriever import retrieve  # 그대로

# (중략)

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
            # ✅ 전역/옵션 기반으로 스티칭 여부 결정 (기본 False)
            stitch_by_path=bool(opts.get("stitch_by_path") if "stitch_by_path" in opts else config.STITCH_BY_PATH),
        )
        state["retrieved"] = hits
        return state
    except Exception as e:
        state["error"] = f"retrieval_error: {e}"
        state["retrieved"] = []
        return state