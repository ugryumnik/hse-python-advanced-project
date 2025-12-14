from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.services import RAGService
from .. import get_rag_service


class SourceRequest(BaseModel):
    filename: str
    page: int
    limit: int | None = 5


class ChunkModel(BaseModel):
    filename: str
    page: int | None
    text: str


class SourceResponse(BaseModel):
    chunks: list[ChunkModel]


source_router = APIRouter()


@source_router.post("/source", response_model=SourceResponse)
async def get_source_chunk(
    request: SourceRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        # Исправлено: search вместо similarity_search
        docs = await rag_service.agent.vector_store.search(
            query=f"{request.filename} страница {request.page}",
            k=request.limit,
            filter_dict={"filename": request.filename, "page": request.page},
        )

        if not docs:
            # Пробуем без фильтра по странице (может быть None)
            docs = await rag_service.agent.vector_store.search(
                query=f"{request.filename}",
                k=request.limit,
                filter_dict={"filename": request.filename},
            )

        if not docs:
            return SourceResponse(chunks=[])

        chunks = [
            ChunkModel(
                filename=d.metadata.get("filename", request.filename),
                page=d.metadata.get("page"),
                text=d.page_content,
            )
            for d in docs
        ]
        return SourceResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))