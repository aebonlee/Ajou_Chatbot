from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):

    await init_db()


    yield


app = FastAPI(
    title="ICT_11",
    description="미정",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check() -> dict:
    """
    서비스 상태 확인용 테스트 엔드포인트
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )