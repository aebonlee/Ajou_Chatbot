# 도구 / 리트리버

from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.clients.chroma import get_collection
from app.config.settings import settings
from konlpy.tag import Okt

_tokenizer = Okt()
def ko_tokens(x: str): return _tokenizer.morphs(x)

def bm25_from_chroma_docs(chroma: Chroma, k=10) -> BM25Retriever:
    # 컬렉션 전체를 문서화해서 BM25 인덱스 (Elasticsearch도 고려)
    docs = chroma.get(include=["metadatas","documents"])
    texts = docs["documents"]
    metas = docs["metadatas"]
    from langchain.schema.document import Document
    documents = [Document(page_content=t, metadata=m) for t,m in zip(texts, metas)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    r = BM25Retriever.from_documents(chunks, preprocess_func=ko_tokens)
    r.k = k
    return r

def build_ensemble(collection_name: str, k: int = 8):
    vs = get_collection(collection_name)
    vector_r = vs.as_retriever(search_kwargs={"k": k})
    bm25_r = bm25_from_chroma_docs(vs, k=k)
    from langchain.retrievers import EnsembleRetriever
    return EnsembleRetriever(retrievers=[vector_r, bm25_r], weights=[0.6, 0.4])