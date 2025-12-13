from contextlib import asynccontextmanager
from fastapi import FastAPI

from infra.llm import LegalRAGAgent
from core.services import RAGService, IngestionService
from .routes import router


# Глобальные сервисы (инициализируются при старте)
rag_service: RAGService | None = None
ingestion_service: IngestionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global rag_service, ingestion_service
    
    # Startup
    agent = LegalRAGAgent()
    rag_service = RAGService(agent=agent)
    ingestion_service = IngestionService(agent=agent)
    
    # optional
    
    yield
    
    # Shutdown
    if rag_service:
        await rag_service.close()


app = FastAPI(
    title="Legal RAG Bot API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


def get_rag_service() -> RAGService:
    """Получение RAG сервиса"""
    if rag_service is None:
        raise RuntimeError("RAG service not initialized")
    return rag_service


def get_ingestion_service() -> IngestionService:
    """Плучение Ingestion сервиса"""
    if ingestion_service is None:
        raise RuntimeError("Ingestion service not initialized")
    return ingestion_service