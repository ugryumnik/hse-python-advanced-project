from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Form, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from infra.db.database import get_session
from infra.db.user_repository import UserRepository

from core.services import IngestionService
from web import get_ingestion_service


class UploadResponse(BaseModel):
    message: str
    chunks_added: int


upload_router = APIRouter()


@upload_router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    user_id: int = Form(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    session: AsyncSession = Depends(get_session)
):
    repo = UserRepository(session)
    user = await repo.get_by_telegram_id(user_id)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: admin only")
    try:
        chunks_count = await ingestion_service.processFile(file)
        
        return UploadResponse(
            message=f"Успешно индексирован документ '{file.filename}'",
            chunks_added=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))