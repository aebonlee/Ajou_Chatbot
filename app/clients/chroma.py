from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config.settings import settings

_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def get_collection(name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        persist_directory=settings.CHROMA_DIR,
        embedding_function=_embeddings
    )