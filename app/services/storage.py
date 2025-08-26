"""
Chroma 래퍼.
- chromadb PersistentClient/Collection 헬퍼
- add()/get_all() 유틸
"""
import os
from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

# ✅ 기본 임베딩을 bge-m3로
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
    if not ids: return
    metas = [_sanitize_meta(m) for m in metas]
    col.add(ids=ids, documents=docs, metadatas=metas)

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