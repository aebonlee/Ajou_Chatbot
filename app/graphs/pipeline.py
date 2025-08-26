"""
LangGraph 파이프라인 조립/실행:
- 노드: parse_intent -> classify -> need_more -> retrieve -> build_context -> answer -> END
- run_rag_graph(): 호출 편의 헬퍼 (옵션들을 opts로 전달)
"""
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END

from .state import GraphState
from .nodes import node_parse_intent, node_need_more, node_retrieve, node_build_context, node_answer
from .nodes_classify import node_classify
from app.core import config

def build_graph() -> Any:
    g = StateGraph(GraphState)
    g.add_node("parse_intent", node_parse_intent)
    g.add_node("classify", node_classify)
    g.add_node("need_more", node_need_more)
    g.add_node("retrieve", node_retrieve)
    g.add_node("build_context", node_build_context)
    g.add_node("answer", node_answer)

    g.set_entry_point("parse_intent")
    g.add_edge("parse_intent", "classify")

    # classify → fixed-answer면 바로 answer, 아니면 need_more
    def after_classify(s: Dict[str, Any]):
        return "answer" if s.get("skip_rag") else "need_more"

    g.add_conditional_edges("classify", after_classify, {"answer": "answer", "need_more": "need_more"})

    # need_more → (clarify)END or retrieve
    g.add_conditional_edges(
        "need_more",
        lambda s: END if s.get("needs_clarification") else "retrieve",
        {"retrieve": "retrieve", END: END}
    )
    g.add_edge("retrieve", "build_context")
    g.add_edge("build_context", "answer")
    g.add_edge("answer", END)
    return g.compile()

def run_rag_graph(
    *,
    question: str,
    user_id: str = "anonymous",
    persist_dir: str = config.PERSIST_DIR,
    collection: str = config.COLLECTION,
    embedding_model: str = config.EMBEDDING_MODEL,
    topk: int = config.TOPK,
    model_name: str = config.LLM_MODEL,  # ← 기본값은 config에서 .env의 LLM_MODEL
    temperature: float = config.TEMPERATURE,
    max_tokens: int = config.MAX_TOKENS,
    use_llm: bool = True,
    debug: bool = False,
    scope_colleges: Optional[List[str]] = None,
    scope_depts: Optional[List[str]] = None,
    micro_mode: Optional[str] = None,
    assemble_budget_chars: int = 40000,
    max_ctx_chunks: int = 12,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_candidates: int = 30,
) -> Dict[str, Any]:
    graph = build_graph()
    init: GraphState = {
        "question": question,
        "user_id": user_id,
        "context_struct": {},
        "needs_clarification": False,
        "clarification_prompt": None,
        "retrieved": [],
        "context": "",
        "answer": None,
        "llm_answer": None,
        "error": None,
        "category": None,       # ← classify가 채움
        "style_guide": None,    # ← classify가 채움
        "skip_rag": False,      # ← classify가 고정답이면 True
        "must_include": [],     # ← build_context가 채움(카테고리별 엔티티)
        "opts": {
            "persist_dir": persist_dir,
            "collection": collection,
            "embedding_model": embedding_model,
            "topk": topk,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_llm": use_llm,
            "debug": debug,
            "scope_colleges": scope_colleges or [],
            "scope_depts": scope_depts or [],
            "assemble_budget_chars": assemble_budget_chars,
            "max_ctx_chunks": max_ctx_chunks,
            "rerank": rerank,
            "rerank_model": rerank_model,
            "rerank_candidates": rerank_candidates,
        },
    }
    if micro_mode is not None:
        init["opts"]["micro_mode"] = micro_mode

    out: GraphState = graph.invoke(init)  # type: ignore
    hits = out.get("retrieved") or []
    return {
        "question": out.get("question"),
        "answer": out.get("answer"),
        "context": out.get("context"),
        "sources": [h.get("path") or (h.get("metadata") or {}).get("path", "") for h in hits],
        "micro_mode": (init["opts"].get("micro_mode") or "exclude"),
        "error": out.get("error"),
        "clarification_prompt": out.get("clarification_prompt"),
        "llm_answer": out.get("llm_answer"),
    }