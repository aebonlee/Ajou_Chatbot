from fastapi import FastAPI, HTTPException
from app.api.schemas import QueryRequest, QueryResponse
from app.services.answer_service import answer_query
from app.config.settings import settings

app = FastAPI(title="ICT Agent Orchestrator")

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        result = await answer_query(req)
        return result
    except Exception as e:
        raise HTTPException(500, f"internal error: {e}")