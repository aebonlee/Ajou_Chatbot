from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.retriever import get_enhanced_filter

# 확인할 DB 디렉토리
PERSIST_DIR = "storage/chroma-notice"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 연결
vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# 현재 포함된 컬렉션 보기
print("컬렉션 목록:", vs._client.list_collections())

# 샘플 문서 3개 확인
print(vs.get(limit=20))

result = get_enhanced_filter("신소재 학사공지")
print(result)