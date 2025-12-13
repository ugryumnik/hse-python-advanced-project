from pathlib import Path
from typing import Any

from infra.llm import (
    LegalRAGAgent,
    RAGConfig,
    YandexGPTConfig,
    QdrantConfig,
)


class RAGService:
    """Сервис для RAG-запросов к юридическим документам"""

    def __init__(
        self,
        agent: LegalRAGAgent | None = None,
        config: RAGConfig | None = None,
        yandex_config: YandexGPTConfig | None = None,
        qdrant_config: QdrantConfig | None = None,
    ):
        """
        Инициализация RAG сервиса.
        
        Можно передать готового агента или конфигурации для создания нового.
        """
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

    def query(self, question: str, k: int | None = None) -> tuple[str, list[dict]]:
        """
        Выполнить RAG-запрос.
        
        Args:
            question: Вопрос пользователя
            k: Количество документов для контекста
            
        Returns:
            Кортеж (ответ, список источников)
        """
        response = self._agent.query(question, k=k)
        return response.answer, response.sources

    def add_document(self, file_path: str | Path) -> int:
        """
        Добавить документ в индекс.
        
        Returns:
            Количество созданных чанков
        """
        return self._agent.add_document(file_path)

    def index_all(self, force: bool = False) -> int:
        """Индексировать все документы из директории"""
        return self._agent.index_documents(force_reindex=force)

    def get_stats(self) -> dict[str, Any]:
        """Статистика системы"""
        return self._agent.get_stats()

    def health_check(self) -> bool:
        """Проверка работоспособности"""
        return self._agent.health_check()

    def close(self) -> None:
        """Закрыть соединения"""
        self._agent.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()