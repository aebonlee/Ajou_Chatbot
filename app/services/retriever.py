"""
하이브리드 검색기:
- BM25(한국어 강화 토크나이저) + Dense(Chroma: bge-m3)
- micro_mode 감지/후필터
- 후보 풀 크게 → Cross-Encoder 재랭크(bge-reranker) → (옵션) path 스티칭
- 쿼리 보강: 마이크로전공/목록성 키워드 자동 추가
- 안정 가드: 안전 정규화, 가중치 클램핑, 후보 하한, 빈 결과 핸들링
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import math
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict

from .storage import get_client, get_collection, get_all
from .textutil import tokenize_ko, normalize_numbers, path_of
from app.core import config  # ✅ 전역 스위치 사용

EPS = 1e-9

# -----------------------------
# 마이크로 모드 감지 / 쿼리 보강
# -----------------------------
def detect_micro_mode(q: str) -> str:
    s = q.replace(" ", "")
    if "마이크로전공제외" in s or "마이크로제외" in s:
        return "exclude"
    if "마이크로전공만" in s or "마이크로만" in s:
        return "only"
    if ("마이크로전공" in s) or ("마이크로포함" in s) or ("마이크로 포함" in s):
        return "include"
    return "exclude"

def _augment_query(q: str) -> str:
    s = q
    low = q.lower()
    if ("마이크로전공" in q) or ("마이크로 전공" in q) or ("micro" in low):
        s += " 마이크로전공 목록 전공명"
    if any(k in q for k in ["요건", "정리", "졸업", "교육과정"]):
        s += " 졸업요건 교육과정"
    return s

# -----------------------------
# 스코어 정규화 유틸 (안전판)
# -----------------------------
def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= 0.0:
        return {k: 0.0 for k in scores}
    span = vmax - vmin
    if span < EPS:
        return {k: 1.0 for k in scores}
    return {k: (v - vmin) / span for k, v in scores.items()}

# -----------------------------
# 개별 검색기
# -----------------------------
def _bm25_rank(scope_idx: List[int], all_docs: List[str], query: str, topn: int) -> Dict[int, float]:
    if not scope_idx or topn <= 0:
        return {}
    corpus = [tokenize_ko(all_docs[i] or "") for i in scope_idx]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize_ko(query)
    if not q_tokens:
        return {}
    scores = bm25.get_scores(q_tokens)
    top_pairs = sorted(zip(scope_idx, scores), key=lambda x: x[1], reverse=True)[:topn]
    return {gi: float(s) for gi, s in top_pairs}

def _dense(col, all_ids: List[str], where: Optional[Dict[str, Any]], q: str, ndense: int) -> Dict[int, float]:
    if ndense <= 0:
        return {}
    try:
        res = col.query(
            query_texts=[q],
            n_results=int(ndense),
            where=where or None,
            include=["distances", "documents", "metadatas"],
        )
    except Exception:
        return {}

    ids = res.get("ids", [[]])[0] if res else []
    dists = res.get("distances", [[]])[0] if res else []
    out: Dict[int, float] = {}
    for cid, d in zip(ids, dists):
        try:
            gi = all_ids.index(cid)
        except ValueError:
            continue
        sim = 1.0 - float(d)
        if not math.isfinite(sim):
            sim = 0.0
        sim = max(0.0, min(1.0, sim))
        out[gi] = max(out.get(gi, 0.0), sim)
    return out

# -----------------------------
# 메타 필터/부스팅
# -----------------------------
def _is_micro(meta: Dict[str, Any]) -> bool:
    v = meta.get("is_micro")
    if v in ("Y", "N"):
        return v == "Y"
    path = (meta.get("path") or "")
    major = (meta.get("major") or "")
    return ("마이크로전공" in path) or ("마이크로전공" in major)

def _section_boost(meta: Dict[str, Any]) -> float:
    sec = (meta.get("section") or "")
    return 2.0 if ("졸업요건" in sec or "교육과정" in sec) else 0.0

def _scope_boost(meta: Dict[str, Any], depts: Optional[List[str]]) -> float:
    return 2.0 if (depts and meta.get("dept") in set(depts)) else 0.0

# -----------------------------
# 재랭크 (Cross-Encoder)
# -----------------------------
def _apply_cross_encoder_rerank(
    question: str,
    candidates: List[Tuple[int, float]],
    all_docs: List[str],
    all_metas: List[Dict[str, Any]],
    model_name: str,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        if debug:
            print("[Retriever] sentence-transformers 미설치 → 재랭크 생략")
        return candidates

    if not candidates:
        return candidates

    ce = CrossEncoder(model_name)
    idxs = [gi for gi, _ in candidates]
    pairs = []
    for gi in idxs:
        meta = all_metas[gi] or {}
        path = (meta.get("path") or "")
        text = (all_docs[gi] or "")
        pairs.append((question, f"{path}\n{text[:1500]}"))

    try:
        ce_scores = ce.predict(pairs)
    except Exception:
        if debug:
            print("[Retriever] CrossEncoder 예측 실패 → 재랭크 생략")
        return candidates

    first = {gi: sc for gi, sc in candidates}
    blended = [(gi, 0.7 * float(cs) + 0.3 * float(first.get(gi, 0.0))) for gi, cs in zip(idxs, ce_scores)]
    blended.sort(key=lambda x: x[1], reverse=True)
    if debug:
        print(f"[Retriever] rerank applied with {model_name}, candidates={len(idxs)}")
    return blended

# -----------------------------
# 경로 기준 스티칭 (옵션)
# -----------------------------
def _stitch_by_path(
    ordered_pairs: List[Tuple[int, float]],
    *,
    all_ids: List[str],
    all_docs: List[str],
    all_metas: List[Dict[str, Any]],
    topk: int,
) -> List[Dict[str, Any]]:
    by_path: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for gi, sc in ordered_pairs:
        raw_doc = all_docs[gi] or ""
        meta = all_metas[gi] or {}
        p = path_of(raw_doc) or meta.get("path", "") or ""
        if not p:
            p = f"__ID__:{all_ids[gi]}"
        by_path[p].append((gi, sc))

    groups_sorted = sorted(by_path.items(), key=lambda kv: max(s for _, s in kv[1]), reverse=True)

    hits: List[Dict[str, Any]] = []
    for path, items in groups_sorted:
        items.sort(key=lambda x: x[1], reverse=True)
        gi_rep, best_score = items[0]
        bodies: List[str] = []
        for gi, _ in items:
            raw = all_docs[gi] or ""
            if raw.startswith("[PATH]") and "\n" in raw:
                raw = raw.split("\n", 1)[1]
            bodies.append(raw.strip())
        stitched_doc = f"[PATH] {path}\n" + "\n\n".join(bodies)

        hits.append({
            "id": all_ids[gi_rep],
            "score": float(best_score),
            "document": stitched_doc,
            "metadata": all_metas[gi_rep],
            "path": path,
        })
        if len(hits) >= topk:
            break
    return hits

# -----------------------------
# 스코프 1개 단위 검색
# -----------------------------
def _single_scope_retrieve(
    question: str, *, persist_dir: str, collection: str, embedding_model: str,
    topk: int, lex_weight: float, scope_colleges: Optional[List[str]],
    scope_depts: Optional[List[str]], micro_mode: str, debug: bool,
    rerank: bool = True, rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidates: int = 40,
    stitch_by_path: bool = False,   # ✅ 추가
) -> List[Dict[str, Any]]:
    client = get_client(persist_dir)
    col = get_collection(client, collection, embedding_model)
    all_ids, all_docs, all_metas = get_all(col)

    where = {"dept": {"$in": scope_depts}} if scope_depts else ({"college": {"$in": scope_colleges}} if scope_colleges else None)

    q_aug = _augment_query(question)
    qn = normalize_numbers(q_aug)

    if scope_depts:
        scope_idx = [i for i, m in enumerate(all_metas) if (m or {}).get("dept") in set(scope_depts)]
    elif scope_colleges:
        scope_idx = [i for i, m in enumerate(all_metas) if (m or {}).get("college") in set(scope_colleges)]
    else:
        scope_idx = list(range(len(all_docs)))

    if not scope_idx:
        return []

    pool = max(int(rerank_candidates or 0), topk * 4, 60, 1)

    bm25_raw = _bm25_rank(scope_idx, all_docs, qn, topn=min(pool, len(scope_idx)))
    bm25 = _normalize(bm25_raw)

    dense_raw = _dense(col, all_ids, where, qn, ndense=min(pool, len(all_ids)))
    dense = _normalize(dense_raw)

    a = 0.85 if lex_weight is None else float(lex_weight)
    if not math.isfinite(a):
        a = 0.85
    a = max(0.0, min(1.0, a))
    b = 1.0 - a

    pool_idx = set(list(bm25.keys()) + list(dense.keys()))
    if not pool_idx:
        return []

    combined: List[Tuple[int, float]] = []
    bm_has = any(v > 0 for v in bm25.values())
    de_has = any(v > 0 for v in dense.values())
    if not bm_has and de_has:
        a, b = 0.0, 1.0
    elif bm_has and not de_has:
        a, b = 1.0, 0.0

    for gi in pool_idx:
        base = a * bm25.get(gi, 0.0) + b * dense.get(gi, 0.0)
        if not math.isfinite(base):
            base = 0.0
        combined.append((gi, base))
    combined.sort(key=lambda x: x[1], reverse=True)

    if rerank and combined:
        cand_n = max(int(rerank_candidates or 0), 1)
        cand = combined[: min(cand_n, len(combined))]
        reranked = _apply_cross_encoder_rerank(
            question, cand, all_docs, all_metas, model_name=rerank_model, debug=debug
        )
        cand_set = {gi for gi, _ in cand}
        others = [(gi, sc) for gi, sc in combined if gi not in cand_set]
        combined = reranked + others

    # micro 후필터
    filtered: List[Tuple[int, float]] = []
    for gi, sc in combined:
        meta = all_metas[gi] or {}
        if micro_mode == "exclude" and _is_micro(meta):
            continue
        if micro_mode == "only" and not _is_micro(meta):
            continue
        filtered.append((gi, sc))

    if not filtered:
        return []

    # 재가중 & 정렬
    rescored: List[Tuple[int, float]] = []
    for gi, sc in filtered:
        meta = all_metas[gi] or {}
        sc2 = sc + _scope_boost(meta, scope_depts) + _section_boost(meta)
        if micro_mode == "only" and _is_micro(meta):
            sc2 += 1.0
        if micro_mode == "exclude" and _is_micro(meta):
            sc2 -= 1.0
        if not math.isfinite(sc2):
            sc2 = 0.0
        rescored.append((gi, sc2))
    rescored.sort(key=lambda x: x[1], reverse=True)

    # ✅ 스티칭 옵션
    if stitch_by_path:
        hits = _stitch_by_path(
            rescored,
            all_ids=all_ids,
            all_docs=all_docs,
            all_metas=all_metas,
            topk=topk,
        )
    else:
        # 스티칭 없이 개별 청크 그대로 반환
        hits = []
        for gi, sc in rescored[:topk]:
            meta = all_metas[gi] or {}
            hits.append({
                "id": all_ids[gi],
                "score": float(sc),
                "document": all_docs[gi],
                "metadata": meta,
                "path": meta.get("path", ""),
            })

    if debug:
        paths = [h.get("path") for h in hits[:5]]
        print(f"[Retriever] stitch={stitch_by_path}  hits={len(hits)}  top paths: {paths}")
    return hits

# -----------------------------
# 복수 스코프 → 병합
# -----------------------------
def retrieve(
    question: str,
    *,
    persist_dir: str,
    collection: str,
    embedding_model: str,
    topk: int = 8,
    lex_weight: float = 0.85,
    scope_colleges: Optional[List[str]] = None,
    scope_depts: Optional[List[str]] = None,
    micro_mode: Optional[str] = None,
    debug: bool = False,
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidates: int = 40,
    stitch_by_path: bool = False,   # ✅ 추가
) -> List[Dict[str, Any]]:
    mm = micro_mode or detect_micro_mode(question)

    if scope_depts and len(scope_depts) > 1:
        per: List[List[Dict[str, Any]]] = []
        for d in scope_depts:
            per.append(_single_scope_retrieve(
                question,
                persist_dir=persist_dir,
                collection=collection,
                embedding_model=embedding_model,
                topk=topk,
                lex_weight=lex_weight,
                scope_colleges=None,
                scope_depts=[d],
                micro_mode=mm,
                debug=debug,
                rerank=rerank,
                rerank_model=rerank_model,
                rerank_candidates=rerank_candidates,
                stitch_by_path=stitch_by_path,   # ✅ 전달
            ))
        scores: Dict[str, float] = {}
        by_id: Dict[str, Dict[str, Any]] = {}
        for docs in per:
            for rank, h in enumerate(docs):
                hid = h["id"]; by_id[hid] = h
                scores[hid] = scores.get(hid, 0.0) + 1.0 / (60 + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [by_id[_id] for _id, _ in ranked]

    return _single_scope_retrieve(
        question,
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        topk=topk,
        lex_weight=lex_weight,
        scope_colleges=scope_colleges,
        scope_depts=scope_depts,
        micro_mode=mm,
        debug=debug,
        rerank=rerank,
        rerank_model=rerank_model,
        rerank_candidates=rerank_candidates,
        stitch_by_path=stitch_by_path,   # ✅ 전달
    )