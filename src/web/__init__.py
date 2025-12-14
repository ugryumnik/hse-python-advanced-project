from contextlib import asynccontextmanager
from fastapi import FastAPI

from infra.llm import LegalRAGAgent
from core.services import RAGService, IngestionService, DocumentGenerationService

# Глобальные сервисы
rag_service: RAGService | None = None
ingestion_service: IngestionService | None = None
doc_generation_service: DocumentGenerationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global rag_service, ingestion_service, doc_generation_service

    # Startup
    agent = LegalRAGAgent()
    rag_service = RAGService(agent=agent)
    ingestion_service = IngestionService(agent=agent)
    doc_generation_service = DocumentGenerationService(agent=agent)

    yield

    # Shutdown
    if rag_service:
        await rag_service.close()


app = FastAPI(
    title="Legal RAG Bot API",
    version="1.0.0",
    lifespan=lifespan
)


def get_rag_service() -> RAGService:
    if rag_service is None:
        raise RuntimeError("RAG service not initialized")
    return rag_service


def get_ingestion_service() -> IngestionService:
    if ingestion_service is None:
        raise RuntimeError("Ingestion service not initialized")
    return ingestion_service


def get_doc_generation_service() -> DocumentGenerationService:
    if doc_generation_service is None:
        raise RuntimeError("Document generation service not initialized")
    return doc_generation_service


from .routes import router

app.include_router(router)