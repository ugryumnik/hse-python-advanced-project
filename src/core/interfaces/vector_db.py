"""Интерфейс для векторного хранилища"""

from abc import ABC, abstractmethod
from typing import List, Any

from langchain_core.documents import Document


class IVectorDB(ABC):
    """Абстрактный интерфейс для векторного хранилища"""

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Добавить документы в хранилище"""
        pass

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[Document]:
        """Универсальный поиск по запросу"""
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict | None = None,
    ) -> List[Document]:
        """Поиск по сходству с опциональной фильтрацией"""
        pass

    @abstractmethod
    async def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
    ) -> List[Document]:
        """MMR поиск для разнообразия результатов"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Количество документов в коллекции"""
        pass

    @abstractmethod
    async def clear_collection(self) -> None:
        """Очистить коллекцию"""
        pass

    @abstractmethod
    async def get_info(self) -> dict[str, Any]:
        """Получить информацию о коллекции"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Закрыть соединение"""
        pass