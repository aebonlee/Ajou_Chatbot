from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import node_parse_intent, node_need_more, node_retrieve, node_build_context, node_answer
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





################공지사항#####################
    
# -------------------------------
# 프롬프트 템플릿 정의
# -------------------------------
template = """
당신은 대학 공지사항을 정확하게 검색하여 제공하는 친절한 AI 비서입니다.
아래에 제공된 "문서" 내용을 바탕으로 사용자의 "질문"에 답변하세요.

**답변 형식:**
- 답변은 찾은 공지사항 리스트를 아래 형식에 맞춰서 제공해야 합니다.
- 만약 여러 개의 공지사항이 있다면, 모두 이 형식으로 리스트업해주세요.
- 각 공지사항은 다음 형식을 따라야 합니다:
[제목]: [공지사항의 제목]
[URL]: [공지사항의 URL]

**특별 지시:**
1. 만약 제공된 "문서"에 사용자의 질문과 관련된 내용이 전혀 없다면, "죄송합니다. 제공된 문서에는 해당 정보가 없습니다."라고 답하세요.
2. 사용자가 특정 학과나 단과대학, 또는 공지 유형(예: 장학)을 언급했지만, 관련 문서가 검색되지 않았을 경우, 사용자에게 해당 정보를 다시 명확하게 물어보세요.
   - 예시 질문: "어떤 학과의 공지사항을 찾으시나요?" 또는 "어떤 종류의 공지를 찾으시나요?"

---
문서:
{context}

---
질문: {question}

답변:
"""
prompt = ChatPromptTemplate.from_template(template)

# 입력 전처리 (question + filter 동시 생성)
# -------------------------------
def enrich_inputs(x):
    return {
        "question": x["question"],
        "filter": get_enhanced_filter(x["question"])
    }


#  RAG 체인 구축
llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NOTICE)

rag_chain = (
    RunnableLambda(enrich_inputs)
    | {
        "context": RunnableLambda(lambda d: dynamic_retriever(d["question"], d["filter"])),
        "question": RunnableLambda(lambda d: d["question"]),
      }
    | prompt
    | llm
    | StrOutputParser()
)

