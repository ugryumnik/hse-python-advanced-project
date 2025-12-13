from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    user_id: int


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


ask_router = APIRouter()


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        return AskResponse(
            answer="Бебра",
            sources=[{"filename": "Bebra", "page": 52}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
