"""
하이브리드 검색 + 섹션 확장:
- BM25 + Dense + (옵션) Cross-Encoder 재랭크
- 후보 1~N개 path/section_id를 골라 **섹션 전체 청크**를 확장/정렬(order_key) → 스티칭
- 질문에 학년/학기 감지 시, 메타 필터로 바로 해당 term 섹션 전체 확장
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import unicodedata
import math
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi

from .storage import get_client, get_collection, get_all, get_where_all
from .textutil import tokenize_ko, normalize_numbers, detect_year_semester_in_query

EPS = 1e-9

# ---------------- 스코어 정규화 ----------------
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

# ---------------- 개별 검색기 ----------------
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
            include=["distances", "metadatas"],
        )
    except Exception:
        return {}
    ids_list = res.get("ids") or []
    dists_list = res.get("distances") or []
    if not ids_list or not dists_list:
        return {}
    ids = ids_list[0] if isinstance(ids_list[0], (list, tuple)) else ids_list
    dists = dists_list[0] if isinstance(dists_list[0], (list, tuple)) else dists_list
    if not ids or not dists:
        return {}
    id2gi = {cid: gi for gi, cid in enumerate(all_ids)}
    out: Dict[int, float] = {}
    for cid, d in zip(ids, dists):
        gi = id2gi.get(cid)
        if gi is None:
            continue
        try:
            sim = 1.0 - float(d)
        except Exception:
            sim = 0.0
        if not math.isfinite(sim):
            sim = 0.0
        sim = max(0.0, min(1.0, sim))
        prev = out.get(gi, 0.0)
        if sim > prev:
            out[gi] = sim
    return out

# ---------------- 재랭크 ----------------
def _apply_cross_encoder_rerank(
    question: str,
    candidates: List[Tuple[int, float]],
    all_docs: List[str],
    all_metas: List[Dict[str, Any]],
    model_name: str,
    debug: bool = False
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

# ---------------- 스티칭/클린업 유틸 ----------------
_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


def _clean_unicode(s: str) -> str:
    """제로폭/비정규 문자 제거, NFKC 정규화, 과도한 개행 압축."""
    if not s:
        return ""
    import re
    import unicodedata

    # 제로폭 문자 제거
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    # NFKC 정규화
    s = unicodedata.normalize("NFKC", s)
    # 과도한 개행 압축 (3개 이상 → 2개)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 앞뒤 공백 제거
    return s.strip()

def _safe_join(parts: List[str]) -> str:
    """오버랩 기반 절단 없이, 파트별 클린업 후 빈 줄 하나로 연결."""
    clean = [_clean_unicode(p) for p in parts if (p or "").strip()]
    return "\n\n".join(clean)

# (참고) 필요 시 남겨두지만 현재는 사용하지 않음.
def _smart_stitch_texts(parts: List[str], overlap_hint: int = 200) -> str:
    if not parts:
        return ""
    out = parts[0] or ""
    for p in parts[1:]:
        a = out[-overlap_hint:]
        b = (p or "")[:overlap_hint]
        cut = 0
        maxk = min(len(a), len(b))
        for k in range(maxk, 0, -1):
            if a[-k:] == b[:k]:
                cut = k
                break
        out = out + (p[cut:] if cut else p)
    return out


def _expand_by_section(col, section_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    동일한 'section'(H2) 또는 동일한 'term'(H4 section_id)에 속한 모든 청크를
    order_key 오름차순으로 모아서 하나로 합친다.
    개선된 스티칭 알고리즘 사용.
    """
    is_term = (section_meta.get("section_id") or "").startswith("term_")
    if is_term:
        where: Dict[str, Any] = {"section_id": section_meta.get("section_id")}
    else:
        # 동일 파일·섹션 범위 필터
        where = {
            "$and": [
                {"source_path": section_meta.get("source_path")},
                {"section": section_meta.get("section")},
                {"college": section_meta.get("college")},
                {"dept": section_meta.get("dept")},
            ]
        }

    ids, docs, metas = get_where_all(col, where)
    metas_docs = list(zip(metas, docs))
    metas_docs.sort(key=lambda md: (md[0].get("order_key") or "999.999.9999"))

    parts = [d or "" for _, d in metas_docs]

    # 개선된 스티칭 사용 - 오버랩된 청크들을 올바르게 연결
    stitched = _smart_stitch_with_overlap_detection(parts, max_overlap=200)

    rep_meta = metas_docs[0][0] if metas_docs else section_meta
    rep_path = rep_meta.get("path") or section_meta.get("path") or ""
    return {
        "id": rep_meta.get("section_id") or rep_meta.get("source") or "",
        "score": 1.0,
        "document": stitched,
        "metadata": rep_meta,
        "path": rep_path,
    }


def _smart_stitch_with_overlap_detection(parts: List[str], max_overlap: int = 200) -> str:
    """
    오버랩 기반 스티칭 개선:
    1. 연속된 청크 간 최대 max_overlap 길이까지 중복 검사
    2. 중복 구간 발견 시 제거하여 연결
    3. 중복이 없으면 단순 연결
    """
    if not parts:
        return ""

    if len(parts) == 1:
        return _clean_unicode(parts[0] or "")

    result = _clean_unicode(parts[0] or "")

    for i in range(1, len(parts)):
        current = _clean_unicode(parts[i] or "")
        if not current:
            continue

        # 이전 결과의 끝부분과 현재 청크의 시작부분에서 중복 검사
        overlap_found = False
        max_check = min(max_overlap, len(result), len(current))

        # 긴 중복부터 짧은 중복까지 확인
        for overlap_len in range(max_check, 10, -1):  # 최소 10자 이상만 중복으로 간주
            if result[-overlap_len:] == current[:overlap_len]:
                # 중복 발견 - 중복 부분 제거하고 연결
                result = result + current[overlap_len:]
                overlap_found = True
                break

        if not overlap_found:
            # 중복 없음 - 단순 연결 (개행으로 구분)
            result = result + "\n\n" + current

    return result

def _top_candidates_with_expand(
    question: str,
    *,
    col,
    all_ids: List[str],
    all_docs: List[str],
    all_metas: List[Dict[str, Any]],
    topk: int,
    lex_weight: float,
    scope_idx: List[int],
    where_dense: Optional[Dict[str, Any]],
    rerank: bool,
    rerank_model: str,
    debug: bool
) -> List[Dict[str, Any]]:
    pool = max(topk * 6, 80)  # 충분히 크게
    qn = normalize_numbers(question)

    bm25_raw = _bm25_rank(scope_idx, all_docs, qn, topn=min(pool, len(scope_idx)))
    bm25 = _normalize(bm25_raw)

    dense_raw = _dense(col, all_ids, where_dense, qn, ndense=min(pool, len(all_ids)))
    dense = _normalize(dense_raw)

    a = 0.85 if lex_weight is None else float(lex_weight)
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
        combined.append((gi, base))
    combined.sort(key=lambda x: x[1], reverse=True)

    if rerank and combined:
        cand_n = min(max(40, topk * 8), len(combined))
        cand = combined[:cand_n]
        combined = _apply_cross_encoder_rerank(question, cand, all_docs, all_metas, rerank_model, debug=debug)

    # 후보 상위에서 path/section_id 기준으로 섹션 확장
    hits: List[Dict[str, Any]] = []
    seen_sections = set()
    for gi, _ in combined:
        meta = all_metas[gi] or {}
        key = meta.get("section_id") or meta.get("path")
        if not key or key in seen_sections:
            continue
        seen_sections.add(key)
        expanded = _expand_by_section(col, meta)
        hits.append(expanded)
        if len(hits) >= topk:
            break
    return hits

# ---------------- 공개 API ----------------
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
    micro_mode: Optional[str] = None,   # 저장되어 있으면 후필터에 사용 가능(여기선 생략)
    debug: bool = False,
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidates: int = 40,        # 유지: 외부 옵션 호환
    stitch_by_path: bool = False,       # 사용하지 않음(섹션 확장 방식으로 대체)
) -> List[Dict[str, Any]]:

    client = get_client(persist_dir)
    col = get_collection(client, collection, embedding_model)
    all_ids, all_docs, all_metas = get_all(col)

    # 범위 스코프
    if scope_depts:
        scope_idx = [i for i, m in enumerate(all_metas) if (m or {}).get("dept") in set(scope_depts)]
        where_dense = {"dept": {"$in": scope_depts}}
    elif scope_colleges:
        scope_idx = [i for i, m in enumerate(all_metas) if (m or {}).get("college") in set(scope_colleges)]
        where_dense = {"college": {"$in": scope_colleges}}
    else:
        scope_idx = list(range(len(all_docs)))
        where_dense = None

    # 1) 질문에서 학년/학기 감지 → term 섹션 직행 확장 시도
    y, s = detect_year_semester_in_query(question)
    if (y or s) and scope_idx:
        base_where: Dict[str, Any] = {}
        if y: base_where["year"] = y
        if s: base_where["semester"] = s
        if scope_depts:
            base_where["dept"] = {"$in": scope_depts}
        elif scope_colleges:
            base_where["college"] = {"$in": scope_colleges}

        _ids_t, _docs_t, metas_t = get_where_all(col, base_where)
        if metas_t:
            # 동일 term(section_id: term_*) 단위로 그룹핑 → 섹션 확장
            by_term: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for m in metas_t:
                sid = m.get("section_id") or ""
                if sid.startswith("term_"):
                    by_term[sid].append(m)
            hits: List[Dict[str, Any]] = []
            for sid, group in by_term.items():
                exp = _expand_by_section(col, group[0])
                hits.append(exp)
                if len(hits) >= topk:
                    break
            if hits:
                if debug:
                    print(f"[Retriever] direct term expand: {y} {s}, hits={len(hits)}")
                return hits

    # 2) 일반 하이브리드 검색 → 상위 후보 섹션 확장
    hits = _top_candidates_with_expand(
        question,
        col=col,
        all_ids=all_ids,
        all_docs=all_docs,
        all_metas=all_metas,
        topk=topk,
        lex_weight=lex_weight,
        scope_idx=scope_idx,
        where_dense=where_dense,
        rerank=rerank,
        rerank_model=rerank_model,
        debug=debug,
    )
    if debug:
        paths = [h.get("path") for h in hits[:min(5, len(hits))]]
        print(f"[Retriever] expanded sections: {paths}")
    return hits