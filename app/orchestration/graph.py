from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
from app.orchestration.intents import Intent, rule_intent
from app.orchestration.tools import build_ensemble
from app.orchestration.rag_chains import make_rag_chain
from app.orchestration.confidence import confidence_from_scores, fallback_message
from app.clients.llm import get_llm

class State(TypedDict):
    question: str
    intent: str
    tool: Optional[str]
    docs: list
    scores: List[float]
    answer: Optional[str]

llm = get_llm()

def classify(state: State):
    # 1) 규칙 우선 → 부족하면 LLM 소분류(생략 가능)
    intent = rule_intent(state["question"]) or Intent.GENERAL
    state["intent"] = intent.value
    return state

# 각 도메인별 리트리버 (컬렉션명은 settings에서)
from app.config.settings import settings
COLL = {
  Intent.ACADEMICS.value: settings.CHROMA_COLLECTION_MAJOR,  # 기본값(전공 요람)
  Intent.MICRO_DH.value: settings.CHROMA_COLLECTION_MICRO_DH,
  Intent.MICRO_IP.value: settings.CHROMA_COLLECTION_MICRO_IP,
  Intent.MICRO_PL.value: settings.CHROMA_COLLECTION_MICRO_PL,
  Intent.NOTICES.value: settings.CHROMA_COLLECTION_NOTICES,
  Intent.TIPS.value: settings.CHROMA_COLLECTION_TIPS,
}

def retrieve(state: State):
    intent = state["intent"]
    if intent == Intent.GENERAL.value:
        # RAG 없이 바로 생성
        chain = make_rag_chain(llm)  # 컨텍스트 없이도 동작 가능하지만 일반 대화 프롬프트를 별도 써도 됨
        state["docs"] = []
        state["scores"] = []
        state["tool"] = "no_rag"
        state["answer"] = llm.invoke(f"다음 질문에 간단히 답해줘: {state['question']}").content
        return state

    collection = COLL[intent]
    ens = build_ensemble(collection, k=8)
    docs = ens.get_relevant_documents(state["question"])
    scores = [d.metadata.get("score", 0.7) for d in docs]  # chroma retriever는 점수 제공 안 할 수 있어 임시값
    state["docs"] = docs
    state["scores"] = scores
    state["tool"] = f"retriever:{collection}"
    return state

def generate(state: State):
    if state["tool"] == "no_rag":
        return state
    chain = make_rag_chain(llm)
    state["answer"] = chain.invoke({"docs": state["docs"], "question": state["question"]})
    return state

def decide(state: State):
    conf = confidence_from_scores(state["scores"])
    state["confidence"] = conf
    if conf < 0.35 and state["tool"] != "no_rag":
        # 폴백: 간단 안내
        state["answer"] = fallback_message(state["question"])
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("decide", decide)
    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "decide")
    g.add_edge("decide", END)
    return g.compile()