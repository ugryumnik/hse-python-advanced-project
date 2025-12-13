"""Интерфейсы репозиториев для работы с БД"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.domain import User, Document, ChatMessage


class IUserRepository(ABC):
    """Интерфейс репозитория пользователей"""

    @abstractmethod
    async def create(self, user: User) -> User:
        """Создать пользователя"""
        pass

    @abstractmethod
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Получить пользователя по ID"""
        pass

    @abstractmethod
    async def get_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """Получить пользователя по Telegram ID"""
        pass

    @abstractmethod
    async def update(self, user: User) -> User:
        """Обновить пользователя"""
        pass

    @abstractmethod
    async def delete(self, user_id: int) -> bool:
        """Удалить пользователя"""
        pass


class IDocumentRepository(ABC):
    """Интерфейс репозитория документов"""

    @abstractmethod
    async def create(self, document: Document) -> Document:
        """Создать запись о документе"""
        pass

    @abstractmethod
    async def get_by_id(self, doc_id: int) -> Optional[Document]:
        """Получить документ по ID"""
        pass

    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """Получить документ по хешу файла"""
        pass

    @abstractmethod
    async def get_by_user(self, user_id: int) -> List[Document]:
        """Получить все документы пользователя"""
        pass

    @abstractmethod
    async def update_status(self, doc_id: int, status: str) -> Optional[Document]:
        """Обновить статус документа"""
        pass

    @abstractmethod
    async def delete(self, doc_id: int) -> bool:
        """Удалить документ"""
        pass


class IChatHistoryRepository(ABC):
    """Интерфейс репозитория истории чата"""

    @abstractmethod
    async def create(self, message: ChatMessage) -> ChatMessage:
        """Создать запись в истории"""
        pass

    @abstractmethod
    async def get_by_user(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChatMessage]:
        """Получить историю пользователя с пагинацией"""
        pass

    @abstractmethod
    async def get_recent(self, user_id: int, limit: int = 10) -> List[ChatMessage]:
        """Получить последние сообщения пользователя"""
        pass

    @abstractmethod
    async def delete_by_user(self, user_id: int) -> int:
        """Удалить всю историю пользователя"""
        pass