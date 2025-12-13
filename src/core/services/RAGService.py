from pathlib import Path
from typing import Any

from infra.llm import (
    LegalRAGAgent,
    RAGConfig,
    YandexGPTConfig,
    QdrantConfig,
)


class RAGService:
    """Асинхронный сервис для RAG-запросов к юридическим документам"""

    def __init__(
        self,
        agent: LegalRAGAgent | None = None,
        config: RAGConfig | None = None,
        yandex_config: YandexGPTConfig | None = None,
        qdrant_config: QdrantConfig | None = None,
    ):
        if agent is not None:
            self._agent = agent
        else:
            self._agent = LegalRAGAgent(
                config=config or RAGConfig(),
                yandex_config=yandex_config or YandexGPTConfig(),
                qdrant_config=qdrant_config or QdrantConfig(),
            )

    @property
    def agent(self) -> LegalRAGAgent:
        return self._agent

    async def query(self, question: str, k: int | None = None) -> tuple[str, list[dict]]:
        """Асинхронный RAG-запрос"""
        response = await self._agent.query(question, k=k)
        return response.answer, response.sources

    async def add_document(self, file_path: str | Path) -> int:
        """Асинхронно добавить документ в индекс"""
        return await self._agent.add_document(file_path)

    async def index_all(self, force: bool = False) -> int:
        """Асинхронно индексировать все документы"""
        return await self._agent.index_documents(force_reindex=force)

    async def get_stats(self) -> dict[str, Any]:
        """Статистика системы"""
        return await self._agent.get_stats()

    async def health_check(self) -> bool:
        """Проверка работоспособности"""
        return await self._agent.health_check()

    async def close(self) -> None:
        """Закрыть соединения"""
        await self._agent.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()