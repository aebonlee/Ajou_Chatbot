from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from .state import GraphState, GraphStateInfo
from .nodes import node_parse_intent, node_need_more, node_retrieve, node_build_context, node_answer, retrieve_node, \
    generate_node, fallback_node, should_generate
from langchain_core.runnables import RunnableConfig
from .nodes_classify import node_classify
from app.core import config
from langchain_core.prompts import ChatPromptTemplate
from app.services.retriever import get_enhanced_filter, dynamic_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from app.core.config import LLM_MODEL_NOTICE

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

    def after_classify(s: Dict[str, Any]):
        return "answer" if s.get("skip_rag") else "need_more"
    g.add_conditional_edges("classify", after_classify, {"answer": "answer", "need_more": "need_more"})

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
    model_name: str = config.LLM_MODEL,
    temperature: float = config.TEMPERATURE,
    max_tokens: int = config.MAX_TOKENS,
    use_llm: bool = True,
    debug: bool = False,
    scope_colleges: Optional[List[str]] = None,
    scope_depts: Optional[List[str]] = None,
    micro_mode: Optional[str] = None,
    assemble_budget_chars: int = 80000,
    max_ctx_chunks: int = 8,
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidates: int = 40,
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
        "category": None,
        "style_guide": None,
        "skip_rag": False,
        "must_include": [],
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



#--------------------------------------------
# 학사공통
# -------------------------------------------

def make_graph():
    graph = StateGraph(GraphStateInfo)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        should_generate,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()

_pipeline_cache = {}

def get_cached_pipeline():
    """그래프 파이프라인을 캐싱하여 반환"""
    if "graph" in _pipeline_cache:
        return _pipeline_cache["graph"]
    app = make_graph()
    _pipeline_cache["graph"] = app
    return app


def route_query_sync(question: str, departments: List[str] = None, selected_list: List[str] = None):
    """
    그래프를 동기적으로 실행하는 함수.
    departments 리스트를 받아 메타데이터 필터링을 수행합니다.
    """
    if departments is None:
        departments = []
    app = get_cached_pipeline()

    inputs = {"question": question, "departments": departments, "user_selected_list": selected_list}

    final_state = app.invoke(inputs)

    return {
        "answer": final_state.get("answer", "오류가 발생했습니다."),
        "documents": final_state.get("documents", [])
    }





################공지사항#####################
    
# -------------------------------
# 프롬프트 템플릿 정의
# -------------------------------
template = """
당신은 아주대학교의 친근하고 도움이 되는 공지사항 안내 도우미입니다.
아래에 제공된 "문서" 내용을 바탕으로 사용자의 "질문"에 친근하게 답변하세요.

**답변 형식:**
- "네, [질문내용]에 대한 공지사항을 찾아드릴게요!" 로 친근하게 시작
- 찾은 공지사항들을 아래 형식으로 정리해서 제공:

📌 **[제목]**
🔗 **링크**: [URL]
📝 **요약**: [주요 내용 1-2줄]

**특별 지시:**
1. 제공된 문서에 관련 공지사항이 있으면 위 형식으로 정리해주세요.
2. 여러 공지가 있으면 모두 📌 아이콘과 함께 나열해주세요.
3. 관련 문서가 전혀 없다면: "죄송합니다. 현재 해당 내용의 공지사항을 찾을 수 없네요. 다른 키워드로 검색해보시거나, 학과 사무실에 직접 문의해보시는 건 어떨까요?"
4. 마지막에 "더 자세한 내용은 위 링크를 통해 확인하실 수 있습니다!" 로 마무리
5. 제목, 링크, 요약은 줄바꿈 후 출력해주세요

---
문서:
{context}

---
질문: {question}

답변:
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    lines = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        title = md.get("title") or ""
        url = md.get("url") or ""

        # 핵심: 실제 문서 내용도 포함
        content = getattr(d, 'page_content', '') or ''

        lines.append(f"- 제목: {title}")
        lines.append(f"- URL: {url}")
        if content:
            lines.append(f"- 내용: {content}")
        lines.append("")  # 빈 줄로 구분

    return "\n".join(lines) if lines else "(검색 결과 없음)"

# 입력 전처리 (question + filter 동시 생성)
# -------------------------------
def enrich_inputs(x):
    f = get_enhanced_filter(x["question"])
    print("[NOTICE] filter =", f)
    return {"question": x["question"], "filter": f}

def _ctx_builder(d):
    docs = dynamic_retriever(d["question"], d["filter"])
    print(f"[NOTICE] retrieved {len(docs)} docs")  # 🔎 개수 확인
    ctx = format_docs(docs)
    print("[NOTICE] context:\n", ctx)              # 🔎 LLM에 주는 문자열
    return {"context": ctx, "question": d["question"]}

#  RAG 체인 구축
llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NOTICE)


rag_chain = (
    RunnableLambda(enrich_inputs)
    | {
        "question": RunnableLambda(lambda d: d["question"]),
        "raw_docs": RunnableLambda(lambda d: dynamic_retriever(d["question"], d["filter"])),
      }
    | RunnableLambda(lambda d: {"question": d["question"], "context": format_docs(d["raw_docs"])})
    | prompt
    | llm
    | StrOutputParser()
)