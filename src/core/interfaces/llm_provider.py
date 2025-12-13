# core/interfaces/llm_provider.py
"""Интерфейс для LLM провайдеров"""

from abc import ABC, abstractmethod
from typing import List


class ILLMProvider(ABC):
    """Абстрактный интерфейс для LLM провайдеров"""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Генерация ответа на основе сообщений"""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Получить эмбеддинг для поискового запроса"""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для списка документов"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Проверка работоспособности"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Закрыть соединения"""
        pass