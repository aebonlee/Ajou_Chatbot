# settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    SLACK_BOT_TOKEN: str
    SLACK_APP_TOKEN: str
    CHROMA_DIR: str = "./app/data/collections"

    CHROMA_COLLECTION_ACADEMICS: str = "academics"
    CHROMA_COLLECTION_MAJOR: str = "major_dm"
    CHROMA_COLLECTION_MICRO_DH: str = "micro_digital_human"
    CHROMA_COLLECTION_MICRO_IP: str = "micro_metaverse_ip"
    CHROMA_COLLECTION_MICRO_PL: str = "micro_metaverse_planning"
    CHROMA_COLLECTION_TIPS: str = "tips"
    CHROMA_COLLECTION_NOTICES: str = "notices"

    # 채널 라우팅용 (Slack 채널 ID를 .env에 넣어놓을 것)
    SLACK_CHANNEL_NOTICES: str | None = None
    SLACK_CHANNEL_QA: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()