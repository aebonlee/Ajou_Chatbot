"""
하이브리드 검색 + 섹션 확장:
- BM25 + Dense + (옵션) Cross-Encoder 재랭크
- 후보 1~N개 path/section_id를 골라 **섹션 전체 청크**를 확장/정렬(order_key) → 스티칭
- 질문에 학년/학기 감지 시, 메타 필터로 바로 해당 term 섹션 전체 확장
"""
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
import re
import math
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
from .storage import get_client, get_collection, get_all, get_where_all
from .textutil import tokenize_ko, normalize_numbers, detect_year_semester_in_query
from .indexer import process_documents
import os
from langchain_community.retrievers import BM25Retriever
from app.core.config import rag_logger, PERSIST_DIR_INFO
from konlpy.tag import Okt
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from app.core.config import EMBEDDING_MODEL
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from datetime import datetime, timedelta
from langchain_chroma import Chroma

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

# =============================================================================
# 학사공통
# =============================================================================

TOKENIZER = Okt()
model_name = "BAAI/bge-m3"

# --- GPU 자동 감지 로직 ---
# torch.cuda.is_available()가 GPU의 존재 여부를 확인합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
rag_logger.info(f"✅ Using device: {device}")
# ---

model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 전역 캐시 딕셔너리 (하나만 존재해야 합니다)
_retriever_cache = {}

def get_filtered_bm25_retriever(all_chunks: List[Document], departments: List[str] = None) -> BM25Retriever:
    """departments에 해당하는 문서들로만 BM25 retriever를 동적으로 생성합니다."""
    if not departments:
        filtered_chunks = all_chunks
    else:
        filtered_chunks = [
            doc for doc in all_chunks
            if doc.metadata.get("source") in departments
        ]

    if not filtered_chunks:
        # 필터링 결과 문서가 없을 경우를 대비해 빈 문서를 담은 retriever 생성
        rag_logger.warning(f"BM25: No documents found for departments: {departments}")
        return BM25Retriever.from_documents(
            [Document(page_content="내용 없음")],
            preprocess_func=lambda x: TOKENIZER.morphs(x)
        )

    bm25_r = BM25Retriever.from_documents(
        filtered_chunks,
        preprocess_func=lambda x: TOKENIZER.morphs(x)
    )
    bm25_r.k = 10
    rag_logger.info(
        f"✅ Dynamically created BM25 retriever with {len(filtered_chunks)} chunks for {departments or 'all documents'}")
    return bm25_r

def get_all_cached_chunks() -> List[Document]:
    """캐시된 전체 문서 청크들을 반환합니다."""
    # 캐시에서 'chunks' 키로 저장된 청크 리스트를 직접 가져옴
    if "chunks" in _retriever_cache:
        return _retriever_cache["chunks"]

    # 캐시가 없다면 retriever 생성을 통해 캐시를 채움
    get_cached_retrievers()
    return _retriever_cache.get("chunks", [])

# === wRRF ===
def weighted_reciprocal_rank_fusion(
        all_retriever_results: List[List[Document]],
        weights: List[float],
        c: int = 60
) -> List[Tuple[Document, float]]:
    content_to_document = {}
    for single_result_list in all_retriever_results:
        for doc in single_result_list:
            if doc.page_content not in content_to_document:
                content_to_document[doc.page_content] = doc

    fused_scores = {}
    for single_result_list, weight in zip(all_retriever_results, weights):
        for rank, doc in enumerate(single_result_list, start=1):
            doc_key = doc.page_content
            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0.0
            score = weight * (1 / (rank + c))
            fused_scores[doc_key] += score

    sorted_results = sorted(fused_scores.items(), key=lambda x: -x[1])
    final_sorted_docs = [(content_to_document[content], score) for content, score in sorted_results]
    return final_sorted_docs

def format_docs(docs, max_chars: int = 1800) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("title", d.metadata.get("source", "doc"))
        page = d.metadata.get("page", "?")
        body = d.page_content[:max_chars].replace("\u200b", "").strip()
        out.append(f"[{i}] {src}, page={page}\n{body}")
    return "\n\n---\n\n".join(out)

def load_chroma(persistent_dir: str, chunks: List) -> Chroma:
    if os.path.exists(persistent_dir) and len(os.listdir(persistent_dir)) > 0:
        rag_logger.info(f"✅ Loading existing ChromaDB from: {persistent_dir}")
        return Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

    rag_logger.info(f"✅ Creating new ChromaDB at: {persistent_dir}")
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persistent_dir)
    return vs

def get_cached_retrievers():
    """
    통합 DB와 전체 청크 리스트를 로드/생성하고 캐시합니다.
    1) storage/chroma-info가 존재하고 비어있지 않으면: 기존 DB 로드 → 청크 재구성
    2) 없으면: MD 파일을 청킹해서 새 DB 생성
    """
    if "unified" in _retriever_cache:
        return _retriever_cache["unified"]

    rag_logger.info("⏳ Building/Loading unified retriever for /info ...")

    # --- 1) 기존 Chroma가 있으면 먼저 로드 ---
    if os.path.isdir(PERSIST_DIR_INFO) and os.listdir(PERSIST_DIR_INFO):
        rag_logger.info(f"✅ Loading existing ChromaDB from: {PERSIST_DIR_INFO}")
        vector_vs = Chroma(persist_directory=PERSIST_DIR_INFO, embedding_function=embeddings)
        chroma_r = vector_vs.as_retriever(search_kwargs={"k": 10})

        # Chroma에서 모든 문서/메타를 꺼내 BM25용 청크 재구성
        try:
            client = get_client(PERSIST_DIR_INFO)
            col = get_collection(client, "langchain", EMBEDDING_MODEL)  # 컬렉션명은 실제 값 사용
            ids, docs, metas = get_all(col)
            all_chunks = [
                Document(page_content=docs[i] or "", metadata=metas[i] or {})
                for i in range(len(ids))
                if (docs[i] or "").strip()
            ]
            if not all_chunks:
                rag_logger.warning("⚠️ Existing Chroma loaded but no documents found; will try to (re)index from MD.")
            else:
                _retriever_cache["chunks"] = all_chunks
                _retriever_cache["unified"] = chroma_r
                rag_logger.info(f"✅ Unified Chroma loaded with {len(all_chunks)} chunks (rebuilt from collection).")
                return chroma_r
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to rebuild chunks from existing Chroma: {e}. Will try reindexing from MD.")

    # --- 2) 여기까지 왔다는 건 DB가 없거나, 청크 재구성이 실패한 경우 → MD로 청킹 ---
    from app.core.config import PDF_FILES
    all_pdf_paths = list(PDF_FILES.values())
    all_chunks = process_documents(all_pdf_paths)

    if not all_chunks:
        # 기존엔 raise로 끊겼음 → 여기서 명확히 안내
        raise ValueError("No chunks were generated from the documents. "
                         "PDF_FILES 경로 또는 2025_*.md 파일 유무를 확인하세요.")

    rag_logger.info(f"✅ Creating new ChromaDB at: {PERSIST_DIR_INFO}")
    vector_vs = Chroma.from_documents(all_chunks, embedding=embeddings, persist_directory=PERSIST_DIR_INFO)
    chroma_r = vector_vs.as_retriever(search_kwargs={"k": 10})
    _retriever_cache["chunks"] = all_chunks
    _retriever_cache["unified"] = chroma_r
    rag_logger.info(f"✅ Unified Chroma index built with {len(all_chunks)} chunks!")
    return chroma_r

#--------------------------------------------
# 공지사항
# -------------------------------------------

# 임베딩 모델 로드
model_name = EMBEDDING_MODEL  # Use the variable from config.py
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# ---- 필드/별칭 매핑: 실제 메타데이터 키에 맞춤 ----
#  * 실제 메타: college_name / department_name / notice_type / date / title / url ...
metadata_mapping = {
    "공과대학": {"college_name": "공과대학", "aliases": ["공대"]},
    "소프트웨어융합대학": {"college_name": "소프트웨어융합대학", "aliases": ["소융대"]},
    "첨단신소재공학과": {"department_name": "첨단신소재공학과", "aliases": ["첨신공", "신소재"]},
    "건설시스템공학과": {"department_name": "건설시스템공학과", "aliases": ["건시공"]},
    "교통시스템공학과": {"department_name": "교통시스템공학과", "aliases": ["교시공"]},
    "소프트웨어학과": {"department_name": "소프트웨어학과", "aliases": ["소프트웨어과","소웨", "소웨과"]},
    "환경안전공학과": {"department_name": "환경안전공학과", "aliases": []},
    "응용화학생명공학과": {"department_name": "응용화학생명공학과", "aliases": ["응화생"]},
    "응용화학공학과": {"department_name": "응용화학공학과", "aliases": ["화공과","화공"]},
    "기계공학과": {"department_name": "기계공학과", "aliases": ["기계과", "꼐"]},
    "건축학과": {"department_name": "건축학과", "aliases": []},
    "산업공학과": {"department_name": "산업공학과", "aliases": ["산공", "산공과"]},
    "융합시스템공학과": {"department_name": "융합시스템공학과", "aliases": ["융시공"]},
    "국방디지털융합학과": {"department_name": "국방디지털융합학과", "aliases": ["국디융"]},
    "디지털미디어학과": {"department_name": "디지털미디어학과", "aliases": ["디미과", "미디어", "미뎌", "미디어학과", "미디어"]},
    "사이버보안학과": {"department_name": "사이버보안학과", "aliases": ["사보"]},
    "인공지능융합학과": {"department_name": "인공지능융합학과", "aliases": ["인공지능과", "인공지능학과", "인지융"]},
    "일반공지": {"category": "일반공지", "aliases": ["일공", "학사공지", "일반 공지"]},
    "장학공지": {"category": "장학공지", "aliases": ["장학", "장학 공지"]}
}

def get_time_filter(query: str) -> dict:
    """자연어 기간(주/달/개월/년) → timestamp 필터. 기본 2주 이내."""
    query = query.lower()
    m = re.search(r"(\d+)\s*(주|달|개월|년)", query)
    if m:
        n = int(m.group(1)); u = m.group(2)
        if "주" in u:
            past = datetime.now() - timedelta(weeks=n)
        elif "달" in u or "개월" in u:
            past = datetime.now() - timedelta(days=n*30)
        elif "년" in u:
            past = datetime.now() - timedelta(days=n*365)
        else:
            past = datetime.now() - timedelta(weeks=2)
    else:
        past = datetime.now() - timedelta(weeks=4)
    return {"date": {"$gte": int(past.timestamp())}}

def _normalize_chroma_where(where: Optional[dict]) -> Optional[dict]:
    """
    Chroma 최상위 where는 단일 조건이면 평문 dict,
    다중이면 {'$and': [ ... ]}. '$and' 배열 길이 1은 금지라 언랩한다.
    """
    if not where:
        return None
    if "$and" in where and isinstance(where["$and"], list) and len(where["$and"]) == 1:
        return where["$and"][0]
    if "$or" in where and isinstance(where["$or"], list) and len(where["$or"]) == 1:
        return where["$or"][0]
    return where

def get_enhanced_filter(query: str) -> Optional[dict]:
    """
    규칙:
      - 조건 0개 → None
      - 조건 1개 → 그 dict 그대로
      - 조건 2개 이상 → {'$and':[...]}
    키는 실제 컬렉션 메타 키(college_name, department_name, notice_type, date) 사용
    """
    query_lower = query.lower()
    conds = []

    # 1) 기간 필터(기본 2주)
    conds.append(get_time_filter(query))

    # 2) 조직/유형 매핑
    for official, meta in metadata_mapping.items():
        terms = [official] + meta.get("aliases", [])
        if any(re.search(t, query_lower) for t in terms):
            # meta 안의 실제 키 1개만 꺼낸다 (college_name / department_name / notice_type 중 하나)
            for key in ("college_name", "department_name", "notice_type"):
                if key in meta:
                    conds.append({key: meta[key]})
                    break
            break

    # 0,1,2+ 케이스 처리
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}

def dynamic_retriever(query: str, filter_dict: Optional[dict]):
    """공지 전용 Chroma 리트리버. 컬렉션명 명시 + where 정규화."""
    from app.core.config import PERSIST_DIR_NOTICE, NOTICE_COLLECTION
    print("[NOTICE] persist:", PERSIST_DIR_NOTICE, "collection:", NOTICE_COLLECTION)
    loaded_vectorstore = Chroma(
        persist_directory=PERSIST_DIR_NOTICE,
        collection_name=NOTICE_COLLECTION,
        embedding_function=embeddings
    )
    where = _normalize_chroma_where(filter_dict)  # 있으면 사용, 없으면 None
    docs = loaded_vectorstore.as_retriever(
        search_kwargs={"filter": where, "k": 5}
    ).invoke(query)

    print(f"[DEBUG] Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"  {i + 1}. {doc.metadata.get('title')}")
        print(f"      Content length: {len(getattr(doc, 'page_content', ''))}")

    return docs