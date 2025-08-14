from pydantic import BaseSettings
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str | None = None
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

    class Config: env_file = ".env"

settings = Settings()