"""
전역 설정/환경변수 모듈.
- .env 로드 후 애플리케이션 전역에서 사용할 기본값을 노출한다.
"""
import os
from dotenv import load_dotenv
from enum import Enum
import logging

load_dotenv()


rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.INFO)
if not rag_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    rag_logger.addHandler(handler)

class CorpusType(Enum):
    RULES = "rules"
    OVERVIEW = "overview"
    CAMPUS_LIFE = "campus_life"

# --- 경로 설정: 프로젝트 루트를 기준으로 모든 경로를 설정합니다. ---
# 이 파일(app/core/config.py)의 위치를 기준으로 프로젝트 루트를 계산합니다. (app/core -> app -> project_root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MD_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "markdown", "academics_common")
MARKDOWN_CACHE_DIR = MD_DIRECTORY # 동일한 경로를 사용
os.makedirs(MARKDOWN_CACHE_DIR, exist_ok=True)
PDF_FILES = {
    CorpusType.RULES: os.path.join(MD_DIRECTORY, "2025_rules.md"),
    CorpusType.OVERVIEW: os.path.join(MD_DIRECTORY, "2025_overview.md"),
    CorpusType.CAMPUS_LIFE: os.path.join(MD_DIRECTORY, "2025_campus_life.md"),
}


# Chroma 저장소 설정
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage/chroma-acad")
PERSIST_DIR_NOTICE = os.getenv("PERSIST_DIR_NOTICE", "storage/chroma-notice")
PERSIST_DIR_INFO = os.getenv("PERSIST_DIR_INFO", "storage/chroma-info")

# 컬렉션
COLLECTION  = os.getenv("COLLECTION", "acad_docs_bge_m3_clean")
NOTICE_COLLECTION = os.getenv("NOTICE_COLLECTION", "langchain")

# 임베딩모델 정의
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# === LLM 설정 ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MODEL_NOTICE = os.getenv("LLM_MODEL_NOTICE", "gemini-1.5-flash")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# 최대 토큰수
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "6144"))

# 검색/결합 파라미터
TOPK = int(os.getenv("TOPK", "8"))
# Lexical/Semantic 가중치 적용
LEX_WEIGHT = float(os.getenv("LEX_WEIGHT", "0.85"))

# Path-level 스티칭 전역 스위치
STITCH_BY_PATH = os.getenv("STITCH_BY_PATH", "false").lower() == "true"

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

def llm_provider_from_model(model_name: str) -> str:
    m = (model_name or "").lower().strip()
    return "anthropic" if m.startswith("claude") else "openai"