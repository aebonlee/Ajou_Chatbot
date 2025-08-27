"""
전역 설정/환경변수 모듈.
- .env 로드 후 애플리케이션 전역에서 사용할 기본값을 노출한다.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Chroma 저장소 설정
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage/chroma-acad")
PERSIST_DIR_NOTICE = os.getenv("PERSIST_DIR", "storage/chroma-notice")

COLLECTION  = os.getenv("COLLECTION", "acad_docs_bge_m3_clean")

# 임베딩모델 정의
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# === LLM 설정 ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MODEL_NOTICE = os.getenv("LLM_MODEL", "gemini-1.5-flash")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# 최대 토큰수
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "6144"))

# 검색/결합 파라미터
TOPK = int(os.getenv("TOPK", "8"))
# Lexical/Semantic 가중치 적용
LEX_WEIGHT = float(os.getenv("LEX_WEIGHT", "0.85"))

# Path-level 스티칭 전역 스위치 (하면 가져오는 문서수 기하급수적으로 늘어나서 기본 OFF)
STITCH_BY_PATH = os.getenv("STITCH_BY_PATH", "false").lower() == "true"

def llm_provider_from_model(model_name: str) -> str:
    m = (model_name or "").lower().strip()
    return "anthropic" if m.startswith("claude") else "openai"