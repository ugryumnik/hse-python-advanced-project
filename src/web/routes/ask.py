from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.services import RAGService
from .. import get_rag_service


class AskRequest(BaseModel):
    query: str
    user_id: int


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


ask_router = APIRouter()


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    try:
        answer, sources = await rag_service.query(request.query)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))