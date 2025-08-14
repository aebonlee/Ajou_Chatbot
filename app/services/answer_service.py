from app.orchestration.graph import build_graph
from app.api.schemas import QueryRequest, QueryResponse

_graph = build_graph()

async def answer_query(req: QueryRequest) -> QueryResponse:
    state = {"question": req.text, "intent":"", "tool":None, "docs":[], "scores":[], "answer":None}
    result = _graph.invoke(state)

    # 간단한 source 요약(있으면)
    sources = []
    for d in (result.get("docs") or [])[:5]:
        sources.append({
            "title": d.metadata.get("title") or d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "meta": d.metadata
        })

    return QueryResponse(
        answer=result["answer"],
        intent=result["intent"],
        tool=result.get("tool"),
        confidence=float(result.get("confidence", 0.0)),
        sources=sources
    )