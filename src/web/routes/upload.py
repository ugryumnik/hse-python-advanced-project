from fastapi import APIRouter, HTTPException, UploadFile, Form, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from infra.db.database import get_session
from infra.db.user_repository import UserRepository
from infra.llm.document_loader import (
    SUPPORTED_EXTENSIONS,
    ARCHIVE_EXTENSIONS,
    COMPOUND_ARCHIVE_EXTENSIONS,
)
from core.services import IngestionService
from web import get_ingestion_service


class ProcessedFileModel(BaseModel):
    filename: str
    chunks: int
    archive_path: str | None = None


class UploadResponse(BaseModel):
    message: str
    chunks_added: int
    file_type: str
    files_processed: int
    processed_files: list[ProcessedFileModel]
    errors: list[str] = []


upload_router = APIRouter()

# Объединяем все поддерживаемые расширения
ALL_SUPPORTED = SUPPORTED_EXTENSIONS | ARCHIVE_EXTENSIONS | COMPOUND_ARCHIVE_EXTENSIONS


def get_file_type(filename: str) -> tuple[bool, str]:
    """Проверить поддержку файла и вернуть тип"""
    filename_lower = filename.lower()

    for ext in COMPOUND_ARCHIVE_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True, "archive"

    for ext in ARCHIVE_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True, "archive"

    for ext in SUPPORTED_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True, "document"

    return False, "unsupported"


@upload_router.post("/upload", response_model=UploadResponse)
async def upload_document(
        file: UploadFile,
        user_id: int = Form(...),
        ingestion_service: IngestionService = Depends(get_ingestion_service),
        session: AsyncSession = Depends(get_session)
):
    """Загрузить и проиндексировать документ или архив."""
    repo = UserRepository(session)
    user = await repo.get_by_telegram_id(user_id)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: admin only")

    filename = file.filename or "unknown"
    is_valid, file_type = get_file_type(filename)

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат файла."
        )

    try:
        result = await ingestion_service.processFile(file)

        message = (
            f"Архив '{filename}' успешно обработан"
            if result.file_type == "archive"
            else f"Документ '{filename}' успешно индексирован"
        )

        return UploadResponse(
            message=message,
            chunks_added=result.chunks_count,
            file_type=result.file_type,
            files_processed=result.files_processed,
            processed_files=[
                ProcessedFileModel(**f) for f in result.processed_files
            ],
            errors=result.errors
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")