from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Form, Depends
from pydantic import BaseModel

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
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    try:
        chunks_count = await ingestion_service.process_file(file)
        
        return UploadResponse(
            message=f"Document '{file.filename}' indexed successfully",
            chunks_added=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))