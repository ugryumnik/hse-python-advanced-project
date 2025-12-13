"""Доменные сущности приложения"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class UserRole(str, Enum):
    """Роли пользователей"""
    USER = "user"
    ADMIN = "admin"


class DocumentStatus(str, Enum):
    """Статусы обработки документа"""
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass
class User:
    """Сущность пользователя"""
    id: int | None = None
    telegram_id: int = 0
    username: str | None = None
    role: UserRole = UserRole.USER
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Document:
    """Сущность документа"""
    id: int | None = None
    filename: str = ""
    file_path: str = ""
    file_hash: str = ""
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    uploaded_by: int | None = None
    status: DocumentStatus = DocumentStatus.PROCESSING


@dataclass
class ChatMessage:
    """Сущность сообщения чата"""
    id: int | None = None
    user_id: int = 0
    query: str = ""
    answer: str = ""
    used_sources: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchResult:
    """Результат поиска в векторной БД"""
    text: str
    filename: str
    page: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)