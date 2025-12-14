from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Form, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from infra.db.database import get_session
from infra.db.user_repository import UserRepository

from core.services import IngestionService
from web import get_ingestion_service

SUPPORTED_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".txt", ".md",
    ".zip", ".tar", ".tgz", ".tbz2", ".txz",
})

COMPOUND_EXTENSIONS = frozenset({".tar.gz", ".tar.bz2", ".tar.xz"})


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


def is_supported_file(filename: str) -> tuple[bool, str]:
    """Проверить поддержку файла"""
    filename_lower = filename.lower()

    for ext in COMPOUND_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True, "archive"

    for ext in SUPPORTED_EXTENSIONS:
        if filename_lower.endswith(ext):
            file_type = "archive" if ext in {".zip", ".tar", ".tgz", ".tbz2", ".txz"} else "document"
            return True, file_type

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
    is_valid, file_type = is_supported_file(filename)

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла."
        )

    try:
        result = await ingestion_service.processFile(file)

        if result.file_type == "archive":
            message = f"Архив '{filename}' успешно обработан"
        else:
            message = f"Документ '{filename}' успешно индексирован"

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