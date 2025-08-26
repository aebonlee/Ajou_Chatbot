"""
CLI 유틸:
- ingest: data/<college>/<dept>/*.md 색인
- query: 검색만 수행(생성X) → 경로/컨텍스트 미리보기
- graphs: LangGraph로 검색+생성까지 수행
"""
import argparse
from app.core import config
from app.services.indexer import index_tree
from app.graphs.pipeline import run_rag_graph

def cmd_ingest(args):
    index_tree(args.root, persist_dir=args.persist, collection=args.collection, embedding_model=args.embedding)

def cmd_query(args):
    out = run_rag_graph(
        question=args.question,
        persist_dir=args.persist,
        collection=args.collection,
        embedding_model=args.embedding,
        topk=args.topk,
        use_llm=False,   # 검색만
        debug=args.debug,
        scope_colleges=args.colleges or None,
        assemble_budget_chars = args.assemble_budget,
        rerank = args.rerank,
        rerank_model = args.rerank_model,
        rerank_candidates = args.rerank_candidates,
    )
    print("=== SOURCES ===")
    for p in out["sources"]:
        print("-", p)
    print("\n=== CONTEXT (snippet) ===")
    print((out["context"] or "")[:1000])

def cmd_graph(args):
    out = run_rag_graph(
        question=args.question,
        persist_dir=args.persist,
        collection=args.collection,
        embedding_model=args.embedding,
        topk=args.topk,
        use_llm=True,
        debug=args.debug,
        scope_colleges=args.colleges or None,
        assemble_budget_chars = args.assemble_budget,
        max_ctx_chunks = args.max_ctx_chunks,
        rerank = args.rerank,
        rerank_model = args.rerank_model,
        rerank_candidates = args.rerank_candidates,
    )
    if out["error"]:
        print("[ERROR]", out["error"])
    elif out.get("clarification_prompt"):
        print("[CLARIFY]", out["clarification_prompt"])
    else:
        print(out["answer"])

def main():
    p = argparse.ArgumentParser("Acad RAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_i = sub.add_parser("ingest", help="Index data/<college>/<dept>/*.md")
    p_i.add_argument("--root", default="data")
    p_i.add_argument("--persist", default=config.PERSIST_DIR)
    p_i.add_argument("--collection", default=config.COLLECTION)
    p_i.add_argument("--embedding", default=config.EMBEDDING_MODEL)
    p_i.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="Retrieve only (no LLM)")
    p_q.add_argument("--question", required=True)
    p_q.add_argument("--persist", default=config.PERSIST_DIR)
    p_q.add_argument("--collection", default=config.COLLECTION)
    p_q.add_argument("--embedding", default=config.EMBEDDING_MODEL)
    p_q.add_argument("--topk", type=int, default=config.TOPK)
    p_q.add_argument("--colleges", nargs="*", default=[])
    p_q.add_argument("--debug", action="store_true")
    p_q.add_argument("--assemble-budget", type=int, default=40000)
    p_q.add_argument("--rerank", action="store_true")
    p_q.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p_q.add_argument("--rerank-candidates", type=int, default=30)
    p_q.set_defaults(func=cmd_query)

    p_g = sub.add_parser("graphs", help="RAG generate (LLM)")
    p_g.add_argument("--question", required=True)
    p_g.add_argument("--persist", default=config.PERSIST_DIR)
    p_g.add_argument("--collection", default=config.COLLECTION)
    p_g.add_argument("--embedding", default=config.EMBEDDING_MODEL)
    p_g.add_argument("--topk", type=int, default=config.TOPK)
    p_g.add_argument("--colleges", nargs="*", default=[])
    p_g.add_argument("--debug", action="store_true")
    p_g.add_argument("--assemble-budget", type=int, default=40000)
    p_g.add_argument("--max-ctx-chunks", type=int, default=12)
    p_g.add_argument("--rerank", action="store_true")
    p_g.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p_g.add_argument("--rerank-candidates", type=int, default=30)
    p_g.set_defaults(func=cmd_graph)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()