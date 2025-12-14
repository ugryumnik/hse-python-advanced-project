from fastapi import APIRouter, HTTPException, UploadFile, Form, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from infra.db.database import get_session
from infra.db.user_repository import UserRepository
from infra.llm.document_loader import SUPPORTED_EXTENSIONS, ARCHIVE_EXTENSIONS
from core.services import IngestionService
from .. import get_ingestion_service


class UploadResponse(BaseModel):
    message: str
    chunks_added: int
    file_type: str
    files_processed: int
    errors: list[str] = []


upload_router = APIRouter()


def get_file_type(filename: str) -> tuple[bool, str]:
    """Проверить поддержку файла"""
    name = filename.lower()
    
    for ext in ARCHIVE_EXTENSIONS:
        if name.endswith(ext):
            return True, "archive"
    
    for ext in SUPPORTED_EXTENSIONS:
        if name.endswith(ext):
            return True, "document"
    
    return False, "unsupported"


@upload_router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    user_id: int = Form(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    session: AsyncSession = Depends(get_session)
):
    """Загрузить и проиндексировать документ или архив"""
    repo = UserRepository(session)
    user = await repo.get_by_telegram_id(user_id)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: admin only")

    filename = file.filename or "unknown"
    is_valid, file_type = get_file_type(filename)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    try:
        result = await ingestion_service.processFile(file)

        return UploadResponse(
            message=f"{'Архив' if result.file_type == 'archive' else 'Документ'} обработан",
            chunks_added=result.chunks_count,
            file_type=result.file_type,
            files_processed=result.files_processed,
            errors=result.errors
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")