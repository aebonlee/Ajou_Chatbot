#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_rag.py
- 하이브리드 리트리벌(BM25 + 벡터) + (옵션) LLM 생성
- '마이크로전공' 질의에 따라 micro_mode 자동 적용
- --dept 여러 개로 학과 스코프 지정 가능

예시 (리트리벌만):
python scripts/test_rag.py \
  --persist storage/chroma-acad \
  --collection acad_docs \
  --embedding sentence-transformers/all-MiniLM-L6-v2 \
  --question "디지털미디어학과/소프트웨어학과 전공필수만 요약해줘" \
  --dept 디지털미디어학과 --dept 소프트웨어학과 \
  --topk 8 --debug

예시 (LLM까지):
export OPENAI_API_KEY=sk-...
python scripts/test_rag.py ... --llm --model gpt-4o-mini
"""
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional

# ---------- 공통 유틸 ----------
def tokenize_ko(text: str) -> List[str]:
    """한글/영문/숫자 토큰 단순화 + 일부 기호 정규화."""
    text = text.lower()
    text = re.sub(r"[·•/／\-\–—]", " ", text)
    text = re.sub(r"[‘’“”‟‚‛′″´`]", "", text)
    tokens = re.findall(r"[가-힣]+|[a-z]+|\d+", text)
    return tokens

def detect_micro_mode(q: str) -> str:
    """질문에 포함된 문구로 마이크로전공 모드 판단."""
    s = q.replace(" ", "")
    if "마이크로전공만" in s or "마이크로만" in s:
        return "only"
    if "마이크로전공제외" in s or "마이크로제외" in s:
        return "exclude"
    if "마이크로전공" in s:
        return "include"
    return "exclude"

def build_hint(q: str) -> str:
    """가벼운 BM25 부스팅 키워드."""
    hints = []
    for kw in ["전공기초", "전공필수", "전공선택", "졸업요건", "교육과정", "권장 이수 순서표"]:
        if kw in q:
            hints.append(kw)
    return " ".join(hints)

# 어떤 리트리벌 결과가 들어와도 동일한 형식(list[dict])으로 정규화
def normalize_hits(hits):
    """
    Returns: list of dicts with keys:
      - document: str
      - metadata: dict
      - score: float
      - path: str
    지원 케이스:
      - list[dict] (이미 'document' 키가 있음)
      - dict (Chroma raw: {"documents":[[...]], "metadatas":[[...]], "distances":[[...]]})
      - list[list[str]] (예: documents[0])
      - list[str]
      - None / 빈 결과
    """
    norm = []
    if not hits:
        return norm

    # case: list[dict] with 'document'
    if isinstance(hits, list) and hits and isinstance(hits[0], dict) and "document" in hits[0]:
        for h in hits:
            meta = h.get("metadata") or {}
            norm.append({
                "document": h.get("document", "") or "",
                "metadata": meta,
                "score": float(h.get("score", 0.0) or 0.0),
                "path": h.get("path") or meta.get("path", "") or "",
            })
        return norm

    # case: Chroma raw dict
    if isinstance(hits, dict) and "documents" in hits:
        docs = (hits.get("documents") or [[]])
        metas = (hits.get("metadatas") or [[]])
        dists = (hits.get("distances") or [[]])
        docs = docs[0] if docs else []
        metas = metas[0] if metas else [{} for _ in range(len(docs))]
        dists = dists[0] if dists else [None for _ in range(len(docs))]
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - float(dist) if dist is not None else 0.0
            meta = meta or {}
            norm.append({
                "document": doc or "",
                "metadata": meta,
                "score": score,
                "path": meta.get("path", "") or "",
            })
        return norm

    # case: list[list[str]] (예: documents[0])
    if isinstance(hits, list) and hits and isinstance(hits[0], list):
        for doc in hits[0]:
            norm.append({"document": doc or "", "metadata": {}, "score": 0.0, "path": ""})
        return norm

    # case: list[str]
    if isinstance(hits, list) and hits and isinstance(hits[0], str):
        for doc in hits:
            norm.append({"document": doc or "", "metadata": {}, "score": 0.0, "path": ""})
        return norm

    return norm

# ---------- 리트리벌 ----------
def retrieve(
    question: str,
    persist_dir: str,
    collection: str,
    embedding_model: str,
    dept_scope: Optional[List[str]] = None,
    topk: int = 8,
    debug: bool = False,
    assemble_budget_chars: int = 20000,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    하이브리드 리트리벌 (BM25+Dense) + 마이크로전공 하드필터 + 섹션 우대 + 부모단위 스티칭.
    반환: (개별 히트 리스트, LLM용 스티칭 컨텍스트)
    """
    import numpy as np
    import chromadb
    from chromadb.utils import embedding_functions
    from rank_bm25 import BM25Okapi

    # ---- load
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    col = client.get_collection(name=collection)
    try:
        col._embedding_function = ef  # type: ignore[attr-defined]
    except Exception:
        pass

    # ---- fetch all (in-memory 인덱스)
    def _get_all(page_size: int = 512):
        ids_all: List[str] = []
        docs_all: List[str] = []
        metas_all: List[Dict[str, Any]] = []
        offset = 0
        while True:
            res = col.get(where={"path": {"$ne": ""}}, include=["documents", "metadatas"], limit=page_size, offset=offset)
            ids = res.get("ids", [])
            if not ids:
                break
            ids_all.extend(ids)
            docs_all.extend(res.get("documents", []))
            metas_all.extend(res.get("metadatas", []))
            if len(ids) < page_size:
                break
            offset += page_size
        return ids_all, docs_all, metas_all

    all_ids, all_docs, all_metas = _get_all()
    if debug:
        print(f"[DEBUG] total docs={len(all_ids)}")

    # ---- 스코프
    if dept_scope:
        dept_set = set(dept_scope)
        scope_idx = [i for i, m in enumerate(all_metas) if m.get("dept") in dept_set]
    else:
        dept_set = set()
        scope_idx = list(range(len(all_docs)))

    # ---- 세부/텀 의도
    def wants_term_granularity(q: str) -> bool:
        s = q.replace(" ", "")
        keys = ["1학년","2학년","3학년","4학년","1학기","2학기","권장이수","권장순서","학기별"]
        return any(k in s for k in keys)

    def is_micro_idx(gi: int) -> bool:
        meta = all_metas[gi] or {}
        if meta.get("is_micro") in ("Y","N"):
            return meta["is_micro"] == "Y"
        maj = (meta.get("major") or "")
        path = (meta.get("path") or "")
        return ("마이크로전공" in maj) or ("마이크로전공" in path)

    # ---- BM25
    corpus = [tokenize_ko(all_docs[i]) for i in scope_idx]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize_ko(question)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_pairs = list(zip(scope_idx, bm25_scores))
    bm25_pairs.sort(key=lambda x: x[1], reverse=True)
    bm25_top = bm25_pairs[: min(64, len(bm25_pairs))]

    # ---- Dense (dept where)
    where = {"dept": {"$in": list(dept_set)}} if dept_set else None
    qres = col.query(
        query_texts=[question],
        n_results=min(64, len(all_ids)),
        where=where,
        include=["distances", "metadatas", "documents"],  # 'ids' 삭제
    )
    ids = qres.get("ids", [[]])[0]
    dists = qres.get("distances", [[]])[0]
    dense_pairs: List[Tuple[int, float]] = []
    for cid, dist in zip(ids, dists):
        try:
            gi = all_ids.index(cid)
        except ValueError:
            continue
        sim = 1.0 - float(dist)
        dense_pairs.append((gi, sim))

    # ---- 정규화
    def _norm(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
        if not pairs:
            return {}
        import numpy as np
        vals = np.array([s for _, s in pairs], dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            return {i: 1.0 for i, _ in pairs}
        return {i: float((s - vmin) / (vmax - vmin)) for i, s in pairs}

    bm25_norm = _norm(bm25_top)
    dense_norm = _norm(dense_pairs)

    # ---- 풀 구성 + 마이크로 하드필터
    pool: List[int] = list(set(list(bm25_norm.keys()) + list(dense_norm.keys())))
    micro_mode = detect_micro_mode(question)
    if micro_mode == "exclude":
        pool = [gi for gi in pool if not is_micro_idx(gi)]
    elif micro_mode == "only":
        pool = [gi for gi in pool if is_micro_idx(gi)]

    # 풀 비었으면 완화 폴백
    if not pool:
        if micro_mode == "include":
            pool = [gi for gi, _ in bm25_top if is_micro_idx(gi)]
        elif micro_mode == "exclude":
            pool = [gi for gi, _ in bm25_top if not is_micro_idx(gi)]
        else:
            pool = [gi for gi, _ in bm25_top]

    # ---- 점수 결합 + 섹션 선호/term 감점
    term_intent = wants_term_granularity(question)
    hits_scored: List[Tuple[int, float]] = []
    for gi in pool:
        meta = all_metas[gi] or {}
        base = 0.5 * bm25_norm.get(gi, 0.0) + 0.5 * dense_norm.get(gi, 0.0)

        # 스코프 가산
        if dept_scope and meta.get("dept") in dept_scope:
            base += 0.2

        # '졸업요건/교육과정' 섹션 가산
        if "졸업요건" in (meta.get("section") or "") or "교육과정" in (meta.get("section") or ""):
            base += 0.15

        # 섹션/텀 가중치
        ctype = (meta.get("chunk_type") or "").lower()
        if ctype == "sec" and not term_intent:
            base += 0.25
        if ctype == "term" and not term_intent:
            base -= 0.10
        if ctype == "term" and term_intent:
            base += 0.10

        hits_scored.append((gi, base))

    hits_scored.sort(key=lambda x: x[1], reverse=True)

    # ---- 경로 중복 제거
    seen_path = set()
    deduped: List[Tuple[int, float]] = []
    for gi, sc in hits_scored:
        path = (all_metas[gi].get("path") or "").strip()
        if path and path in seen_path:
            continue
        seen_path.add(path)
        deduped.append((gi, sc))
        if len(deduped) >= topk:
            break

    # ---- 결과 리스트 (로그/LLM context용)
    results: List[Dict[str, Any]] = []
    for gi, sc in deduped:
        meta = all_metas[gi]
        doc = all_docs[gi]
        path_line = meta.get("path") or ""
        results.append(
            {"id": all_ids[gi], "score": float(sc), "document": doc, "metadata": meta, "path": path_line}
        )

    # ---- 부모 단위 스티칭 (LLM 투입용 컨텍스트)
    order_year = {"1학년":1,"2학년":2,"3학년":3,"4학년":4}
    order_sem  = {"1학기":1,"2학기":2}

    def _sort_key(meta: Dict[str, Any]) -> Tuple[int,int,str]:
        return (order_year.get(meta.get("year",""), 99),
                order_sem.get(meta.get("semester",""), 99),
                meta.get("chunk_type",""))

    stitched_parts: List[str] = []
    used_parents: set = set()
    for item in results:
        meta = item["metadata"] or {}
        pid = meta.get("parent_id") or meta.get("id")
        if not pid or pid in used_parents:
            continue
        siblings: List[int] = [j for j, m in enumerate(all_metas)
                               if (m.get("parent_id") == pid) or (m.get("id") == pid)]
        siblings.sort(key=lambda j: _sort_key(all_metas[j]))
        head = meta.get("path") or ""
        blob = [f"### {head}"]
        for j in siblings:
            blob.append(all_docs[j])
        stitched = "\n\n".join(blob)
        stitched_parts.append(stitched)
        used_parents.add(pid)

        if sum(len(x) for x in stitched_parts) >= assemble_budget_chars:
            break

    assembled_context = "\n\n".join(stitched_parts)
    if len(assembled_context) > assemble_budget_chars:
        assembled_context = assembled_context[:assemble_budget_chars]

    if debug:
        print(f"[DEBUG] micro_mode={micro_mode}")
        top_paths = [r["path"] for r in results]
        print("[DEBUG] top paths:", top_paths)

    return results, assembled_context

def pretty_print_hits(hits):
    """
    정규화된 히트들을 보기 좋게 출력
    - 어떤 형태가 들어와도 normalize_hits()로 통일
    """
    normalized = normalize_hits(hits)
    print("\n=== RETRIEVAL HITS ===")
    if not normalized:
        print("(no hits)")
        return

    for i, h in enumerate(normalized, 1):
        score = h.get("score", 0.0)
        meta = h.get("metadata") or {}
        path = h.get("path", "") or meta.get("path", "")
        body = h.get("document", "") or ""
        print(f"[{i}] score={score:.3f}  PATH: {path}")
        print(f"    META: {meta}")
        print("-----\n")
        print(body[:1200], "\n")

# ---------- LLM 생성 ----------
def generate_answer(question, hits, model="gpt-4o-mini", system_prompt=None, max_ctx_chunks=8, **kwargs):
    """
    LLM 컨텍스트 생성 + 호출
    - hits를 먼저 normalize_hits()로 통일
    - 상위 max_ctx_chunks개만 컨텍스트에 넣음(너무 길어지는 것 방지)
    """
    from openai import OpenAI
    client = OpenAI()

    normalized = normalize_hits(hits)
    if not normalized:
        context = "관련 문서를 찾지 못했습니다."
    else:
        context_parts = []
        for i, h in enumerate(normalized[:max_ctx_chunks], 1):
            path = h.get("path", "") or (h.get("metadata") or {}).get("path", "")
            doc  = h.get("document", "") or ""
            context_parts.append(f"[{i}] PATH: {path}\n{doc}")
        context = "\n\n".join(context_parts)

    sys_msg = system_prompt or (
        "당신은 학사 안내/교육과정 RAG 어시스턴트입니다. "
        "주어진 컨텍스트에서만 답하며, 없으면 '문서에 없음'이라고 말하세요."
    )
    user_msg = f"질문: {question}\n\n---\n컨텍스트:\n{context}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content": sys_msg},
            {"role":"user", "content": user_msg},
        ],
        temperature=kwargs.get("temperature", 0.2),
        max_tokens=kwargs.get("max_tokens", 700),
    )
    return resp.choices[0].message.content.strip()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--question", required=True)
    ap.add_argument("--dept", action="append", default=[], help="학과 스코프 (여러 번 지정 가능)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--llm", action="store_true", help="LLM으로 최종 답변 생성")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=700)
    args = ap.parse_args()

    # retrieve()는 (hits, assembled_context) 튜플 반환 → 언팩 필수
    hits, assembled_context = retrieve(
        question=args.question,
        persist_dir=args.persist,
        collection=args.collection,
        embedding_model=args.embedding,
        dept_scope=args.dept or None,
        topk=args.topk,
        debug=args.debug,
    )

    pretty_print_hits(hits)

    if args.llm:
        print("\n=== LLM ANSWER ===")
        ans = generate_answer(
            question=args.question,
            hits=hits,  # 필요시 assembled_context로 교체 가능
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(ans)

if __name__ == "__main__":
    main()