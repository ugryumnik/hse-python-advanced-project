"""API для генерации документов"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import quote

from core.services import DocumentGenerationService
from infra.db.database import get_session
from infra.db.user_repository import UserRepository
from .. import get_doc_generation_service


class GenerateRequest(BaseModel):
    request: str
    context: str | None = None
    user_id: int
    use_rag: bool = True


class GenerateResponse(BaseModel):
    title: str
    markdown: str
    document_type: str | None


class DocumentTypesResponse(BaseModel):
    types: dict[str, str]


generate_router = APIRouter()


_DOCUMENT_TYPE_LABELS_RU: dict[str, str] = {
    "contract": "Оферта",
}


def _localize_document_types(document_types: dict[str, str]) -> dict[str, str]:
    value_translations = {
        "Offer": "Оферта",
    }

    localized: dict[str, str] = {}
    for doc_type, label in document_types.items():
        localized[doc_type] = _DOCUMENT_TYPE_LABELS_RU.get(
            doc_type,
            value_translations.get(label, label),
        )
    return localized


@generate_router.post("/generate", response_model=GenerateResponse)
async def generate_document(
        request: GenerateRequest,
        doc_service: DocumentGenerationService = Depends(get_doc_generation_service),
        session: AsyncSession = Depends(get_session)
):
    """Сгенерировать документ и вернуть markdown"""
    repo = UserRepository(session)
    user = await repo.get_by_telegram_id(request.user_id)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        result = await doc_service.generate(
            request=request.request,
            context=request.context,
            use_rag=request.use_rag,
        )

        return GenerateResponse(
            title=result.title,
            markdown=result.markdown_content,
            document_type=result.document_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@generate_router.post("/generate/pdf")
async def generate_document_pdf(
        request: GenerateRequest,
        doc_service: DocumentGenerationService = Depends(get_doc_generation_service),
        session: AsyncSession = Depends(get_session)
):
    """Сгенерировать документ и вернуть PDF"""
    repo = UserRepository(session)
    user = await repo.get_by_telegram_id(request.user_id)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        result = await doc_service.generate(
            request=request.request,
            context=request.context,
            use_rag=request.use_rag,
        )

        # Формируем имя файла
        safe_title = "".join(c for c in result.title if c.isalnum() or c in " -_").strip()
        safe_title = safe_title[:50] or "document"
        filename = f"{safe_title}.pdf"
        utf8_filename = quote(filename)

        return Response(
            content=result.pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{utf8_filename}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@generate_router.get("/generate/types", response_model=DocumentTypesResponse)
async def get_document_types(
        doc_service: DocumentGenerationService = Depends(get_doc_generation_service),
):
    """Получить список поддерживаемых типов документов"""
    return DocumentTypesResponse(types=_localize_document_types(doc_service.get_document_types()))