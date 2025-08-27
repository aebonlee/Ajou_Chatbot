"""
Chroma 래퍼.
- chromadb PersistentClient/Collection 헬퍼
- add()/add_with_embeddings()/get_all()/get_where_all() 유틸
- where 강제 $and 래핑으로 Chroma 최상위 연산자 제약 해결
"""
import os
from typing import List, Tuple, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

# 기본 임베딩: bge-m3
DEFAULT_EMBEDDING = "BAAI/bge-m3"

def get_client(persist_dir: str) -> chromadb.PersistentClient:
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def _make_embedding_fn(model_name: str):
    """
    sentence-transformers 로딩. bge-m3 포함 모든 ST 호환 모델 지원.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

def get_collection(client: chromadb.PersistentClient, name: str, embedding_model: str = DEFAULT_EMBEDDING) -> Collection:
    ef = _make_embedding_fn(embedding_model)
    try:
        col = client.get_collection(name=name)
    except Exception:
        # cosine 거리 가정
        col = client.create_collection(
            name=name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        return col
    # attach embedding function (query 시 필요)
    col._embedding_function = ef  # type: ignore[attr-defined]
    return col

_ALLOWED = (bool, int, float, str)
def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, _ALLOWED) else str(v)
    return out

def add(col: Collection, ids: List[str], docs: List[str], metas: List[Dict[str, Any]]) -> None:
    """
    기본 add: 임베딩은 컬렉션 내장 embedding_function으로 계산.
    (clean_text만 넣는 경로가 필요할 때 유지)
    """
    if not ids: return
    metas = [_sanitize_meta(m) for m in metas]
    col.add(ids=ids, documents=docs, metadatas=metas)

def add_with_embeddings(col: Collection, ids: List[str], docs: List[str], metas: List[Dict[str, Any]], embs: List[List[float]]) -> None:
    """
    커스텀 임베딩을 전달하여 저장.
    - docs는 clean_text
    - embs는 '가상 프리픽스 + clean_text'로 계산된 임베딩
    """
    if not ids: return
    metas = [_sanitize_meta(m) for m in metas]
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

def get_all(col: Collection, page_size: int = 512) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids_all: List[str] = []
    docs_all: List[str] = []
    metas_all: List[Dict[str, Any]] = []
    offset = 0
    while True:
        res = col.get(
            where={"path": {"$ne": ""}},   # no-op 유사 필터
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids = res.get("ids", [])
        if not ids: break
        ids_all.extend(ids)
        docs_all.extend(res.get("documents", []))
        metas_all.extend(res.get("metadatas", []))
        if len(ids) < page_size: break
        offset += page_size
    return ids_all, docs_all, metas_all

# --------- where 강제 $and 래핑 유틸 ---------
def _force_and(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chroma where는 최상위에 연산자 하나만 허용한다.
    규칙:
      - None -> None
      - 이미 $and/$or 로 시작하면 그대로 둔다
      - 평문 dict 에서 '키가 1개'면 그대로 둔다 (래핑하지 않음)
      - 평문 dict 에서 '키가 2개 이상'이면 {"$and":[{k:v}, ...]} 로 래핑
    """
    if not where:
        return None
    # 이미 논리연산자면 그대로
    if any(str(k).startswith("$") for k in where.keys()):
        return where
    # 단일 조건은 그대로 사용 (Chroma: $and 최소 2개 필요)
    if len(where) <= 1:
        return where
    # 다중 조건만 $and로 래핑
    return {"$and": [{k: v} for k, v in where.items()]}

def get_where_all(col: Collection, where: Optional[Dict[str, Any]], page_size: int = 512) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    where 조건으로 전량 페이지네이션 조회.
    - 최상위 where는 항상 _force_and 로 안전하게 래핑
    """
    ids_all: List[str] = []
    docs_all: List[str] = []
    metas_all: List[Dict[str, Any]] = []
    offset = 0
    safe_where = _force_and(where)
    while True:
        res = col.get(
            where=safe_where,
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids = res.get("ids", [])
        if not ids: break
        ids_all.extend(ids)
        docs_all.extend(res.get("documents", []))
        metas_all.extend(res.get("metadatas", []))
        if len(ids) < page_size: break
        offset += page_size
    return ids_all, docs_all, metas_all